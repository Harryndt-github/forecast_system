"""
==============================================
LLM FINE-TUNER - Domain-Specific Fine-Tuning
==============================================
Fine-tune local LLM (Qwen2.5) trên forecast data thực tế.

Approach: QLoRA (Quantized Low-Rank Adaptation)
- Tạo training dataset từ historical predictions vs actuals
- Fine-tune LoRA adapters (~50MB, không modify model gốc)
- Merge adapter vào GGUF model cho inference

Architecture:
    ┌─────────────────────────────────────────────┐
    │           LLM Fine-Tuner                    │
    │                                             │
    │   1. Generate Training Data (from Master)   │
    │   2. Create Instruction Dataset             │
    │   3. Fine-tune with QLoRA                   │
    │   4. Export LoRA adapter → GGUF             │
    │   5. Validate fine-tuned model              │
    └─────────────────────────────────────────────┘

Self-Learning Loop:
    historical_data → training_pairs → fine-tune → better_predictions
    → new_actuals → update_training → re-fine-tune (weekly/monthly)

Requirements:
    pip install peft transformers datasets accelerate bitsandbytes trl
"""

import json
import os
import re
import time
import datetime
import hashlib
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from forecast_system.config.settings import (
    PROJECT_ROOT, CURRENT_DATE, RAG_LLM_CONFIG,
    MASTER_FILE_NAME,
)
from forecast_system.utils.logger import get_logger

logger = get_logger('llm_finetuner')

# ==========================================
# DEPENDENCY CHECKS
# ==========================================

HAS_TORCH = False
HAS_TRANSFORMERS = False
HAS_PEFT = False
HAS_TRL = False
HAS_DATASETS = False
HAS_BNB = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, BitsAndBytesConfig,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    from peft import (  # type: ignore[import-not-found]
        LoraConfig, get_peft_model, prepare_model_for_kbit_training,
        PeftModel, TaskType,
    )
    HAS_PEFT = True
except ImportError:
    pass

try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  # type: ignore[import-not-found]
    HAS_TRL = True
except ImportError:
    pass

try:
    from datasets import Dataset  # type: ignore[import-not-found]
    HAS_DATASETS = True
except ImportError:
    pass

try:
    import bitsandbytes  # type: ignore[import-not-found]
    HAS_BNB = True
except ImportError:
    pass


# ==========================================
# FINE-TUNE CONFIG
# ==========================================

FINETUNE_DIR = str(PROJECT_ROOT / 'finetune')
TRAINING_DATA_DIR = str(PROJECT_ROOT / 'finetune' / 'training_data')
ADAPTER_DIR = str(PROJECT_ROOT / 'finetune' / 'adapters')
FINETUNED_GGUF_DIR = str(PROJECT_ROOT / 'finetune' / 'finetuned_models')

# Default fine-tuning settings
DEFAULT_FINETUNE_CONFIG = {
    # LoRA parameters
    'lora_r': 16,                   # LoRA rank (higher = more capacity, more memory)
    'lora_alpha': 32,               # LoRA alpha (scaling factor)
    'lora_dropout': 0.05,           # Dropout for LoRA layers
    'target_modules': [             # Modules to apply LoRA
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],
    
    # Training parameters
    'num_epochs': 3,                # Training epochs
    'batch_size': 4,                # Batch size (adjust for VRAM)
    'gradient_accumulation': 4,     # Effective batch = batch_size × accumulation
    'learning_rate': 2e-4,          # Learning rate
    'warmup_ratio': 0.1,            # Warmup proportion
    'max_seq_length': 2048,         # Max sequence length (shorter = faster)
    'weight_decay': 0.01,           # Weight decay
    'lr_scheduler': 'cosine',       # LR scheduler type
    'fp16': True,                   # Use FP16 (if supported)
    'bf16': False,                  # Use BF16 (if supported)
    
    # Data generation
    'min_training_pairs': 100,      # Minimum training examples
    'max_training_pairs': 5000,     # Maximum training examples
    'train_split_ratio': 0.9,       # Train/eval split
    'lookback_days': 60,            # Use last N days of actuals for training
    
    # Validation
    'eval_steps': 50,               # Evaluate every N steps
    'save_steps': 100,              # Save checkpoint every N steps
    'logging_steps': 10,            # Log every N steps
    
    # Quantization (for QLoRA)
    'use_4bit': True,               # 4-bit quantization
    'bnb_4bit_compute_dtype': 'float16',
    'bnb_4bit_quant_type': 'nf4',
    'use_nested_quant': False,
}


# ==========================================
# TRAINING DATA GENERATOR
# ==========================================

class TrainingDataGenerator:
    """
    Tạo training dataset cho fine-tuning từ historical predictions.
    
    Mỗi training example là một cặp (prompt, response) trong đó:
    - prompt: history + restaurant context + events (giống RAG prompt)
    - response: actual guest counts (ground truth)
    
    Đây là cách LLM "học" từ data thực tế thay vì chỉ dùng RAG.
    """
    
    @staticmethod
    def generate_from_master_file(
        df_master: pd.DataFrame,
        df_train: pd.DataFrame = None,  # type: ignore[reportArgumentType]
        brain_memory: Dict = None,  # type: ignore[reportArgumentType]
        lookback_days: int = 60,
        max_pairs: int = 5000,
    ) -> List[Dict]:
        """
        Generate training pairs from Master Forecast file.
        
        Mỗi training pair = 1 restaurant × 1 tuần forecast
        - Input: 21-day history + context
        - Output: 7-day actual guest counts
        
        Returns:
            List of {"text": formatted_instruction} dicts
        """
        if df_master is None or df_master.empty:
            logger.warning("No master data for training generation")
            return []
        
        # Prefer raw LLM outputs when available so the fine-tuner learns
        # from the LLM's own mistakes rather than the fully corrected system output.
        prediction_col = (
            'AI_Raw_Daily_Forecast'
            if 'AI_Raw_Daily_Forecast' in df_master.columns
            else 'Final_Predicted_Guests'
        )
        ai_available = (
            df_master['AI_Forecast_Available'].fillna(False).astype(bool)
            if prediction_col == 'AI_Raw_Daily_Forecast' and 'AI_Forecast_Available' in df_master.columns
            else True
        )

        # Filter: only rows with both prediction and actual
        mask = (
            pd.notna(df_master.get(prediction_col)) &  # type: ignore[reportOperatorIssue]
            pd.notna(df_master.get('Actual_Guest')) &
            (df_master['Actual_Guest'] >= 0) &
            ai_available
        )
        df = df_master[mask].copy()
        
        if len(df) < 100:
            logger.warning(f"Not enough data for training: {len(df)} rows (need ≥100)")
            return []
        
        # Convert dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date  # type: ignore[reportAttributeAccessIssue]
        
        # Filter by lookback
        cutoff = CURRENT_DATE - datetime.timedelta(days=lookback_days)
        df = df[df['Date'] >= cutoff]
        
        if len(df) < 100:
            logger.warning(f"Not enough recent data: {len(df)} rows after {lookback_days}d filter")
            return []
        
        # Aggregate to daily level
        daily = df.groupby(['Restaurant_Code', 'Date']).agg({  # type: ignore[reportAttributeAccessIssue]
            'Actual_Guest': 'sum',
            prediction_col: 'first' if prediction_col == 'AI_Raw_Daily_Forecast' else 'sum',
            'Weekday': 'first',
            'Is_Holiday': 'first',
            'Is_Veg': lambda x: x.any() if hasattr(x, 'any') else False,
        }).reset_index()
        daily = daily.rename(columns={prediction_col: 'Model_Predicted_Guests'})
        
        # Sort
        daily = daily.sort_values(['Restaurant_Code', 'Date'])
        
        training_pairs = []
        restaurants = daily['Restaurant_Code'].unique()
        
        logger.info(f"Generating training data from {len(restaurants)} restaurants...")
        
        for res_code in restaurants:
            res_data = daily[daily['Restaurant_Code'] == res_code].sort_values('Date')  # type: ignore[reportCallIssue]
            
            if len(res_data) < 14:  # Need at least 14 days
                continue
            
            # Generate sliding window pairs
            pairs = TrainingDataGenerator._generate_sliding_window_pairs(
                res_data, str(res_code), brain_memory
            )
            training_pairs.extend(pairs)
            
            if len(training_pairs) >= max_pairs:
                break
        
        # Shuffle and cap
        np.random.shuffle(training_pairs)
        training_pairs = training_pairs[:max_pairs]
        
        logger.info(
            f"Generated {len(training_pairs)} training pairs "
            f"from {len(restaurants)} restaurants"
        )
        
        return training_pairs
    
    @staticmethod
    def _generate_sliding_window_pairs(
        res_data: pd.DataFrame,
        res_code: str,
        brain_memory: Dict = None,  # type: ignore[reportArgumentType]
    ) -> List[Dict]:
        """
        Generate sliding window training pairs from 1 restaurant.
        
        Window: 14 days history → 7 days forecast
        Stride: 3 days (overlap for more data)
        """
        pairs = []
        history_len = 14
        forecast_len = 7
        stride = 3
        
        dates = res_data['Date'].values
        actuals = res_data['Actual_Guest'].values
        weekdays = res_data['Weekday'].values
        holidays = res_data['Is_Holiday'].values
        
        for start_idx in range(0, len(dates) - history_len - forecast_len + 1, stride):
            hist_end = start_idx + history_len
            fc_end = hist_end + forecast_len
            
            # History window
            hist_dates = dates[start_idx:hist_end]
            hist_actuals = actuals[start_idx:hist_end]
            hist_weekdays = weekdays[start_idx:hist_end]
            hist_holidays = holidays[start_idx:hist_end]
            
            # Forecast window (ground truth)
            fc_dates = dates[hist_end:fc_end]
            fc_actuals = actuals[hist_end:fc_end]
            fc_weekdays = weekdays[hist_end:fc_end]
            fc_holidays = holidays[hist_end:fc_end]
            
            # Build training text
            text = TrainingDataGenerator._format_training_example(
                res_code=res_code,
                hist_dates=hist_dates,
                hist_actuals=hist_actuals,
                hist_weekdays=hist_weekdays,
                hist_holidays=hist_holidays,
                fc_dates=fc_dates,
                fc_actuals=fc_actuals,
                fc_weekdays=fc_weekdays,
                fc_holidays=fc_holidays,
                brain_memory=brain_memory,
            )
            
            pairs.append({
                "text": text,
                "restaurant_code": res_code,
                "history_end": pd.Timestamp(hist_dates[-1]).strftime('%Y-%m-%d'),
                "forecast_start": pd.Timestamp(fc_dates[0]).strftime('%Y-%m-%d'),
                "forecast_end": pd.Timestamp(fc_dates[-1]).strftime('%Y-%m-%d'),
                "target_total": int(np.sum(fc_actuals)),  # type: ignore[reportArgumentType]
            })
        
        return pairs
    
    @staticmethod
    def _format_training_example(
        res_code: str,
        hist_dates, hist_actuals, hist_weekdays, hist_holidays,
        fc_dates, fc_actuals, fc_weekdays, fc_holidays,
        brain_memory: Dict = None,  # type: ignore[reportArgumentType]
    ) -> str:
        """
        Format một training example theo Instruction format.
        
        Format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
        (Qwen2.5 ChatML format)
        """
        # System prompt (shorter version for training efficiency)
        system = (
            "You are a Demand Planner for Vietnamese restaurants. "
            "Given historical guest counts and events, predict daily total guests. "
            "Output JSON array only: [{\"date\": \"YYYY-MM-DD\", \"forecast\": N}, ...]"
        )
        
        # User prompt: history + targets
        user_parts = [f"Restaurant: {res_code}\n"]
        
        # Brain context (if available)
        if brain_memory:
            res_mem = brain_memory.get('restaurants', {}).get(res_code, {})
            if res_mem:
                bias = res_mem.get('overall_bias', 0)
                if abs(bias) > 2:
                    direction = "over" if bias > 0 else "under"
                    user_parts.append(
                        f"NOTE: This model tends to {direction}-predict by ~{abs(bias):.0f} guests.\n"
                    )
        
        # History
        user_parts.append("HISTORY (Date | Weekday | Guest Count):")
        for i in range(len(hist_dates)):
            d = pd.Timestamp(hist_dates[i]).strftime('%Y-%m-%d')  # type: ignore[reportAttributeAccessIssue]
            wd = str(hist_weekdays[i])
            cnt = int(hist_actuals[i])
            hol = " HOLIDAY" if hist_holidays[i] else ""
            user_parts.append(f"- {d} ({wd}): {cnt}{hol}")
        
        user_parts.append("")
        
        # Forecast targets
        user_parts.append("FORECAST TARGETS (Date | Weekday | Event):")
        for i in range(len(fc_dates)):
            d = pd.Timestamp(fc_dates[i]).strftime('%Y-%m-%d')  # type: ignore[reportAttributeAccessIssue]
            wd = str(fc_weekdays[i])
            evt = "HOLIDAY" if fc_holidays[i] else "Normal"
            if wd in ['Saturday', 'Sunday']:
                evt = "WEEKEND" if evt == "Normal" else f"{evt} | WEEKEND"
            user_parts.append(f"- {d} ({wd}) : {evt}")
        
        user_parts.append("\nGenerate JSON forecast:")
        user = "\n".join(user_parts)
        
        # Assistant response (ground truth)
        response_items = []
        for i in range(len(fc_dates)):
            d = pd.Timestamp(fc_dates[i]).strftime('%Y-%m-%d')  # type: ignore[reportAttributeAccessIssue]
            cnt = int(fc_actuals[i])
            response_items.append(f'    {{"date": "{d}", "forecast": {cnt}}}')
        
        assistant = "[\n" + ",\n".join(response_items) + "\n]"
        
        # Format as ChatML (Qwen2.5 format)
        text = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>"
        )
        
        return text

    @staticmethod
    def save_training_data(
        training_pairs: List[Dict],
        output_dir: str = None,  # type: ignore[reportArgumentType]
    ) -> str:
        """Save training data to JSONL file."""
        output_dir = output_dir or TRAINING_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(output_dir, f'training_data_{timestamp}.jsonl')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in training_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        logger.info(f"Training data saved: {filepath} ({len(training_pairs)} examples)")
        return filepath
    
    @staticmethod
    def load_training_data(filepath: str) -> List[Dict]:
        """Load training data from JSONL file."""
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs


# ==========================================
# LLM FINE-TUNER (QLoRA)
# ==========================================

class LLMFineTuner:
    """
    Fine-tune local LLM trên forecast data bằng QLoRA.
    
    QLoRA = Quantized LoRA:
    - Load model ở 4-bit quantization (giảm VRAM ~4x)
    - Train chỉ LoRA adapter layers (~50MB)
    - Merge adapter + base model → model mới
    
    Usage:
        finetuner = LLMFineTuner()
        
        # Step 1: Generate training data
        training_data = TrainingDataGenerator.generate_from_master_file(df_master)
        
        # Step 2: Fine-tune
        result = finetuner.finetune(training_data)
        
        # Step 3: Export adapter
        finetuner.export_adapter()
    """
    
    def __init__(self, config: Dict = None):  # type: ignore[reportArgumentType]
        self.config = {**DEFAULT_FINETUNE_CONFIG, **(config or {})}
        self._model = None
        self._tokenizer = None
        self._adapter_path = None
        
        # Detect device capabilities
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect available hardware for training."""
        self.device_info = {
            'has_cuda': False,
            'has_mps': False,
            'gpu_name': 'CPU',
            'gpu_memory_gb': 0,
        }
        
        if HAS_TORCH:
            if torch.cuda.is_available():  # type: ignore[reportPossiblyUnboundVariable]
                self.device_info['has_cuda'] = True
                self.device_info['gpu_name'] = torch.cuda.get_device_name(0)  # type: ignore[reportPossiblyUnboundVariable]
                mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)  # type: ignore[reportPossiblyUnboundVariable]
                self.device_info['gpu_memory_gb'] = round(mem, 1)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # type: ignore[reportPossiblyUnboundVariable]
                self.device_info['has_mps'] = True
                self.device_info['gpu_name'] = 'Apple Silicon (MPS)'
                # Apple Silicon shared memory, estimate from system
                try:
                    import subprocess
                    result = subprocess.run(
                        ['sysctl', '-n', 'hw.memsize'],
                        capture_output=True, text=True
                    )
                    total_ram_gb = int(result.stdout.strip()) / (1024**3)
                    # MPS can use ~75% of system RAM
                    self.device_info['gpu_memory_gb'] = round(total_ram_gb * 0.75, 1)
                except Exception:
                    self.device_info['gpu_memory_gb'] = 8
        
        logger.info(f"Hardware: {self.device_info['gpu_name']} "
                    f"({self.device_info['gpu_memory_gb']}GB)")
    
    def check_requirements(self) -> Dict:
        """
        Check if all fine-tuning requirements are met.
        
        Returns:
            Dict with 'ready', 'missing', 'warnings'
        """
        result = {
            'ready': True,
            'missing': [],
            'warnings': [],
            'hardware': self.device_info,
        }
        
        # Check required libraries
        if not HAS_TORCH:
            result['missing'].append('torch (pip install torch)')
            result['ready'] = False
        if not HAS_TRANSFORMERS:
            result['missing'].append('transformers (pip install transformers)')
            result['ready'] = False
        if not HAS_PEFT:
            result['missing'].append('peft (pip install peft)')
            result['ready'] = False
        if not HAS_TRL:
            result['missing'].append('trl (pip install trl)')
            result['ready'] = False
        if not HAS_DATASETS:
            result['missing'].append('datasets (pip install datasets)')
            result['ready'] = False
        
        # Check model
        base_model = self._get_base_model_name()
        if not base_model:
            result['missing'].append(
                'Base model (need HuggingFace model name, e.g. Qwen/Qwen2.5-1.5B-Instruct)'
            )
            result['ready'] = False
        
        # Hardware warnings
        if not self.device_info['has_cuda'] and not self.device_info['has_mps']:
            result['warnings'].append(
                'No GPU detected. Training will be very slow on CPU. '
                'Consider using Google Colab or a machine with GPU.'
            )
        
        if self.device_info['has_mps']:
            result['warnings'].append(
                'Apple Silicon MPS detected. Using CPU training with Metal acceleration. '
                'bitsandbytes 4-bit quantization may not work on MPS - falling back to FP16.'
            )
            # Disable 4-bit for MPS (not fully supported)
            self.config['use_4bit'] = False
        
        if self.device_info['gpu_memory_gb'] < 8 and self.device_info['has_cuda']:
            result['warnings'].append(
                f'Low GPU memory ({self.device_info["gpu_memory_gb"]}GB). '
                'Using smaller batch size and gradient accumulation.'
            )
            self.config['batch_size'] = 1
            self.config['gradient_accumulation'] = 8
        
        return result
    
    def _get_base_model_name(self) -> Optional[str]:
        """
        Get the HuggingFace model name for fine-tuning.
        
        We use the Qwen2.5-1.5B-Instruct for fine-tuning (smaller, faster).
        The GGUF files are for inference only; fine-tuning needs HF format.
        """
        # Default: use 1.5B model for fine-tuning (faster, less VRAM)
        model_name = os.getenv(
            'FINETUNE_MODEL_NAME',
            'Qwen/Qwen2.5-1.5B-Instruct'
        )
        return model_name
    
    def finetune(
        self,
        training_data: List[Dict],
        output_dir: str = None,  # type: ignore[reportArgumentType]
        resume_from: str = None,  # type: ignore[reportArgumentType]
    ) -> Dict:
        """
        Fine-tune LLM on forecast data using QLoRA.
        
        Args:
            training_data: List of {"text": formatted_instruction} dicts
            output_dir: Directory to save adapter (default: ADAPTER_DIR)
            resume_from: Path to resume training from checkpoint
            
        Returns:
            Dict with training results and adapter path
        """
        # Check requirements
        req_check = self.check_requirements()
        if not req_check['ready']:
            logger.error(f"Missing requirements: {req_check['missing']}")
            return {
                'status': 'FAILED',
                'reason': f"Missing: {', '.join(req_check['missing'])}",
            }
        
        for warn in req_check['warnings']:
            logger.warning(f"⚠️ {warn}")
        
        if len(training_data) < self.config['min_training_pairs']:
            logger.warning(
                f"Not enough training data: {len(training_data)} "
                f"(need ≥ {self.config['min_training_pairs']})"
            )
            return {
                'status': 'SKIPPED',
                'reason': f'Need ≥{self.config["min_training_pairs"]} examples, got {len(training_data)}',
            }
        
        output_dir = output_dir or ADAPTER_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("🎯 Starting LLM Fine-Tuning (QLoRA)")
        logger.info(f"   Training examples: {len(training_data)}")
        logger.info(f"   Base model: {self._get_base_model_name()}")
        logger.info(f"   LoRA rank: {self.config['lora_r']}")
        logger.info(f"   Epochs: {self.config['num_epochs']}")
        logger.info(f"   Output: {output_dir}")
        logger.info("=" * 60)
        
        try:
            # Step 1: Prepare dataset
            logger.info("📊 Step 1: Preparing dataset...")
            train_dataset, eval_dataset = self._prepare_datasets(training_data)
            
            # Step 2: Load base model + tokenizer
            logger.info("📦 Step 2: Loading base model...")
            model, tokenizer = self._load_base_model()
            
            # Step 3: Apply LoRA
            logger.info("🔧 Step 3: Applying LoRA adapters...")
            model = self._apply_lora(model)
            
            # Log trainable params
            trainable, total = self._count_params(model)
            logger.info(
                f"   Trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)"
            )
            
            # Step 4: Training
            logger.info("🏋️ Step 4: Training...")
            training_args = self._get_training_args(output_dir)
            
            trainer = SFTTrainer(  # type: ignore[reportPossiblyUnboundVariable]
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                max_seq_length=self.config['max_seq_length'],
                packing=False,  # Don't pack multiple examples
                dataset_text_field="text",
            )
            
            # Train
            train_result = trainer.train(
                resume_from_checkpoint=resume_from
            )
            
            # Step 5: Save adapter
            logger.info("💾 Step 5: Saving LoRA adapter...")
            adapter_path = os.path.join(output_dir, 'forecast_adapter')
            trainer.save_model(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            
            self._adapter_path = adapter_path
            
            # Step 6: Save training metadata
            elapsed = time.time() - start_time
            metadata = {
                'status': 'SUCCESS',
                'timestamp': datetime.datetime.now().isoformat(),
                'base_model': self._get_base_model_name(),
                'adapter_path': adapter_path,
                'training_examples': len(training_data),
                'train_split': len(train_dataset),
                'eval_split': len(eval_dataset),
                'epochs': self.config['num_epochs'],
                'lora_r': self.config['lora_r'],
                'lora_alpha': self.config['lora_alpha'],
                'training_loss': train_result.training_loss if hasattr(train_result, 'training_loss') else None,
                'elapsed_seconds': round(elapsed, 1),
                'hardware': self.device_info,
                'config': {k: str(v) for k, v in self.config.items()},
            }
            
            meta_path = os.path.join(adapter_path, 'finetune_metadata.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("=" * 60)
            logger.info(f"✅ Fine-tuning complete!")
            logger.info(f"   Duration: {elapsed/60:.1f} minutes")
            logger.info(f"   Training loss: {metadata['training_loss']}")
            logger.info(f"   Adapter saved: {adapter_path}")
            logger.info("=" * 60)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            traceback.print_exc()
            return {
                'status': 'FAILED',
                'reason': str(e),
                'elapsed_seconds': time.time() - start_time,
            }
    
    def _prepare_datasets(self, training_data: List[Dict]) -> Tuple:
        """Split training data into chronological train/eval datasets."""
        sorted_data = sorted(
            training_data,
            key=lambda item: (
                str(item.get('forecast_start', '')),
                str(item.get('restaurant_code', '')),
            ),
        )
        split_idx = int(len(sorted_data) * self.config['train_split_ratio'])
        split_idx = max(1, min(split_idx, len(sorted_data) - 1)) if len(sorted_data) > 1 else len(sorted_data)

        train_data = sorted_data[:split_idx]
        eval_data = sorted_data[split_idx:]
        
        train_dataset = Dataset.from_list(train_data)  # type: ignore[reportPossiblyUnboundVariable]
        eval_dataset = Dataset.from_list(eval_data) if eval_data else None  # type: ignore[reportPossiblyUnboundVariable]
        
        logger.info(f"   Train: {len(train_data)}, Eval: {len(eval_data)}")
        return train_dataset, eval_dataset
    
    def _load_base_model(self):
        """Load base model with optional quantization."""
        model_name = self._get_base_model_name()
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[reportPossiblyUnboundVariable]
            model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        
        # Quantization config
        if self.config['use_4bit'] and HAS_BNB and self.device_info['has_cuda']:
            logger.info("   Using 4-bit quantization (QLoRA)")
            quant_config = BitsAndBytesConfig(  # type: ignore[reportPossiblyUnboundVariable]
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config['bnb_4bit_quant_type'],
                bnb_4bit_compute_dtype=getattr(
                    torch, self.config['bnb_4bit_compute_dtype']  # type: ignore[reportPossiblyUnboundVariable]
                ),
                bnb_4bit_use_double_quant=self.config['use_nested_quant'],
            )
            model = AutoModelForCausalLM.from_pretrained(  # type: ignore[reportPossiblyUnboundVariable]
                model_name,  # type: ignore[reportArgumentType]
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)  # type: ignore[reportPossiblyUnboundVariable]
        else:
            # FP16/FP32 (for MPS or CPU)
            dtype = torch.float16 if self.device_info['has_cuda'] else torch.float32  # type: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]
            
            device_map = "auto"
            if self.device_info['has_mps']:
                device_map = {"": "mps"}
                dtype = torch.float16  # type: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]
            elif not self.device_info['has_cuda']:
                device_map = {"": "cpu"}
            
            logger.info(f"   Loading model in {dtype} on {device_map}")
            model = AutoModelForCausalLM.from_pretrained(  # type: ignore[reportPossiblyUnboundVariable]
                model_name,  # type: ignore[reportArgumentType]
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        
        model.config.use_cache = False  # Required for gradient checkpointing
        
        self._model = model
        self._tokenizer = tokenizer
        
        return model, tokenizer
    
    def _apply_lora(self, model):
        """Apply LoRA to the model."""
        lora_config = LoraConfig(  # type: ignore[reportPossiblyUnboundVariable]
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=self.config['target_modules'],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,  # type: ignore[reportPossiblyUnboundVariable]
        )
        
        model = get_peft_model(model, lora_config)  # type: ignore[reportPossiblyUnboundVariable]
        return model
    
    def _count_params(self, model) -> Tuple[int, int]:
        """Count trainable vs total parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total
    
    def _get_training_args(self, output_dir: str):
        """Build training arguments."""
        # Adjust for hardware
        fp16 = self.config['fp16'] and self.device_info['has_cuda']
        bf16 = self.config['bf16'] and self.device_info['has_cuda']
        
        # For MPS, use FP16 via torch
        if self.device_info['has_mps']:
            fp16 = False
            bf16 = False
        
        args = TrainingArguments(  # type: ignore[reportPossiblyUnboundVariable]
            output_dir=output_dir,
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            warmup_ratio=self.config['warmup_ratio'],
            lr_scheduler_type=self.config['lr_scheduler'],
            
            # Precision
            fp16=fp16,
            bf16=bf16,
            
            # Evaluation & saving
            eval_strategy="steps" if self.config.get('eval_steps') else "no",
            eval_steps=self.config.get('eval_steps', 50),
            save_strategy="steps",
            save_steps=self.config.get('save_steps', 100),
            save_total_limit=2,
            
            # Logging
            logging_steps=self.config.get('logging_steps', 10),
            logging_dir=os.path.join(output_dir, 'logs'),
            report_to="none",  # No wandb/mlflow
            
            # Optimization
            gradient_checkpointing=True,
            optim="adamw_torch",
            max_grad_norm=0.3,
            
            # Determinism
            seed=42,
            
            # Disable FSDP for single-GPU
            ddp_find_unused_parameters=False,
        )
        
        return args
    
    def export_to_gguf(
        self,
        adapter_path: str = None,  # type: ignore[reportArgumentType]
        output_path: str = None,  # type: ignore[reportArgumentType]
    ) -> Optional[str]:
        """
        Merge LoRA adapter with base model and export to GGUF.
        
        This creates a fine-tuned GGUF model that can be loaded by llama-cpp.
        
        Args:
            adapter_path: Path to LoRA adapter (default: last trained)
            output_path: Path for GGUF output
            
        Returns:
            Path to exported GGUF or None
        """
        adapter_path = adapter_path or self._adapter_path  # type: ignore[reportAssignmentType]
        if not adapter_path or not os.path.exists(adapter_path):
            logger.error(f"Adapter not found: {adapter_path}")
            return None
        
        output_dir = output_path or FINETUNED_GGUF_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            logger.info("🔀 Merging LoRA adapter with base model...")
            
            model_name = self._get_base_model_name()
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(  # type: ignore[reportPossiblyUnboundVariable]
                model_name,  # type: ignore[reportArgumentType]
                torch_dtype=torch.float16,  # type: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]
                device_map="cpu",
                trust_remote_code=True,
            )
            
            # Load and merge adapter
            model = PeftModel.from_pretrained(base_model, adapter_path)  # type: ignore[reportPossiblyUnboundVariable]
            model = model.merge_and_unload()
            
            # Save merged model (HuggingFace format)
            merged_path = os.path.join(output_dir, 'merged_model')
            model.save_pretrained(merged_path)
            
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)  # type: ignore[reportPossiblyUnboundVariable]
            tokenizer.save_pretrained(merged_path)
            
            logger.info(f"✅ Merged model saved: {merged_path}")
            logger.info(
                "To convert to GGUF, run:\n"
                f"  python llama.cpp/convert_hf_to_gguf.py {merged_path} "
                f"--outfile {output_dir}/forecast-qwen-1.5b-ft.gguf "
                f"--outtype q4_k_m"
            )
            
            return merged_path
            
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            traceback.print_exc()
            return None
    
    @staticmethod
    def get_latest_adapter() -> Optional[str]:
        """Find the latest trained adapter."""
        adapter_path = os.path.join(ADAPTER_DIR, 'forecast_adapter')
        if os.path.exists(adapter_path):
            meta_file = os.path.join(adapter_path, 'finetune_metadata.json')
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, encoding='utf-8') as f:
                        meta = json.load(f)
                    logger.info(
                        f"Found adapter: {adapter_path} "
                        f"(trained: {meta.get('timestamp', 'unknown')}, "
                        f"loss: {meta.get('training_loss', 'N/A')})"
                    )
                except Exception:
                    pass
            return adapter_path
        return None

    @staticmethod
    def get_adapter_health(adapter_path: str = None) -> Dict:
        """Return adapter freshness/quality info for inference gating."""
        adapter_path = adapter_path or os.path.join(ADAPTER_DIR, 'forecast_adapter')
        health = {
            'available': False,
            'ready_for_inference': False,
            'adapter_path': adapter_path,
            'age_days': None,
            'training_loss': None,
            'reason': 'adapter_missing',
        }

        if not os.path.exists(adapter_path):
            return health

        health['available'] = True
        meta_file = os.path.join(adapter_path, 'finetune_metadata.json')
        if not os.path.exists(meta_file):
            health['reason'] = 'metadata_missing'
            return health

        try:
            with open(meta_file, encoding='utf-8') as f:
                meta = json.load(f)

            trained_at = datetime.datetime.fromisoformat(meta['timestamp'])
            age_days = (datetime.datetime.now() - trained_at).days
            loss = meta.get('training_loss')

            health['age_days'] = age_days
            health['training_loss'] = loss

            if age_days > 14:
                health['reason'] = 'stale_adapter'
                return health
            if loss is not None and loss > 2.0:
                health['reason'] = 'training_loss_too_high'
                return health

            health['ready_for_inference'] = True
            health['reason'] = 'ok'
            return health
        except Exception as e:
            health['reason'] = f'metadata_error: {e}'
            return health
    
    @staticmethod
    def should_retrain(
        adapter_path: str = None,  # type: ignore[reportArgumentType]
        max_age_days: int = 7,
    ) -> bool:
        """
        Check if the adapter needs retraining.
        
        Returns True if:
        - No adapter exists
        - Adapter is older than max_age_days
        - Training metadata indicates poor quality
        """
        adapter_path = adapter_path or os.path.join(ADAPTER_DIR, 'forecast_adapter')
        
        if not os.path.exists(adapter_path):
            return True
        
        meta_file = os.path.join(adapter_path, 'finetune_metadata.json')
        if not os.path.exists(meta_file):
            return True
        
        try:
            with open(meta_file, encoding='utf-8') as f:
                meta = json.load(f)
            
            # Check age
            trained_at = datetime.datetime.fromisoformat(meta['timestamp'])
            age_days = (datetime.datetime.now() - trained_at).days
            
            if age_days >= max_age_days:
                logger.info(
                    f"Adapter is {age_days} days old (max: {max_age_days}). "
                    f"Retraining recommended."
                )
                return True
            
            # Check quality (if training loss is too high)
            loss = meta.get('training_loss')
            if loss and loss > 2.0:
                logger.info(f"Training loss is high ({loss:.3f}). Retraining recommended.")
                return True
            
            logger.info(f"Adapter is up-to-date ({age_days}d old, loss={loss})")
            return False
            
        except Exception:
            return True


# ==========================================
# FINE-TUNED MODEL LOADER 
# (for integration with RAG agent)
# ==========================================

class FineTunedModelLoader:
    """
    Load fine-tuned model (base + LoRA adapter) cho inference.
    
    Tích hợp vào LocalLLM trong rag_forecast_agent.py:
    - Nếu có adapter → load base_model + adapter → inference
    - Nếu không có → fallback to GGUF model (hiện tại)
    """
    
    _instance = None
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._loaded = False
    
    @classmethod
    def get_instance(cls) -> 'FineTunedModelLoader':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load(self, adapter_path: str = None) -> bool:  # type: ignore[reportArgumentType]
        """
        Load fine-tuned model with LoRA adapter.
        
        Returns True if successful.
        """
        if self._loaded:
            return True
        
        if not HAS_TORCH or not HAS_TRANSFORMERS or not HAS_PEFT:
            logger.debug("Fine-tuned model loader: missing dependencies")
            return False
        
        adapter_path = adapter_path or LLMFineTuner.get_latest_adapter()  # type: ignore[reportAssignmentType]
        if not adapter_path:
            logger.debug("No fine-tuned adapter found")
            return False
        
        try:
            logger.info(f"Loading fine-tuned model from: {adapter_path}")
            
            # Read metadata to get base model name
            meta_file = os.path.join(adapter_path, 'finetune_metadata.json')
            base_model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
            if os.path.exists(meta_file):
                with open(meta_file, encoding='utf-8') as f:
                    meta = json.load(f)
                base_model_name = meta.get('base_model', base_model_name)
            
            # Determine device and dtype
            if torch.cuda.is_available():  # type: ignore[reportPossiblyUnboundVariable]
                dtype = torch.float16  # type: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]
                device_map = "auto"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # type: ignore[reportPossiblyUnboundVariable]
                dtype = torch.float16  # type: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]
                device_map = {"": "mps"}
            else:
                dtype = torch.float32  # type: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]
                device_map = {"": "cpu"}
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(  # type: ignore[reportPossiblyUnboundVariable]
                base_model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            
            # Load adapter
            model = PeftModel.from_pretrained(base_model, adapter_path)  # type: ignore[reportPossiblyUnboundVariable]
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[reportPossiblyUnboundVariable]
                adapter_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self._model = model
            self._tokenizer = tokenizer
            self._loaded = True
            
            logger.info(f"✅ Fine-tuned model loaded (base: {base_model_name})")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned model: {e}")
            return False
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> Optional[str]:
        """Generate response using fine-tuned model."""
        if not self._loaded:
            return None
        
        try:
            # Build ChatML prompt
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            inputs = self._tokenizer(  # type: ignore[reportOptionalCall]
                prompt, return_tensors="pt", truncation=True,
                max_length=2048
            )
            
            # Move to model device
            device = next(self._model.parameters()).device  # type: ignore[reportOptionalMemberAccess]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():  # type: ignore[reportPossiblyUnboundVariable]
                outputs = self._model.generate(  # type: ignore[reportOptionalMemberAccess]
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[reportOptionalMemberAccess]
                )
            
            # Decode only the generated part
            input_len = inputs['input_ids'].shape[1]
            generated = outputs[0][input_len:]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)  # type: ignore[reportOptionalMemberAccess]
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            logger.error(f"Fine-tuned model generation failed: {e}")
            return None
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def unload(self):
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._loaded = False
        if HAS_TORCH:
            if torch.cuda.is_available():  # type: ignore[reportPossiblyUnboundVariable]
                torch.cuda.empty_cache()  # type: ignore[reportPossiblyUnboundVariable]


# ==========================================
# VALIDATION
# ==========================================

class FineTuneValidator:
    """Validate fine-tuned model quality."""
    
    @staticmethod
    def validate_predictions(
        model_loader: FineTunedModelLoader,
        eval_data: List[Dict],
        max_samples: int = 20,
    ) -> Dict:
        """
        Validate fine-tuned model on evaluation data.
        
        Compares predicted forecasts vs ground truth.
        """
        if not model_loader.is_loaded:
            return {'status': 'NOT_LOADED'}
        
        results = {
            'total_samples': 0,
            'parse_success': 0,
            'mape_values': [],
            'mae_values': [],
        }
        
        samples = eval_data[:max_samples]
        
        for sample in samples:
            text = sample.get('text', '')
            if not text:
                continue
            
            # Extract ground truth from assistant response
            try:
                assistant_match = re.search(
                    r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
                    text, re.DOTALL
                )
                if not assistant_match:
                    continue
                
                gt_text = assistant_match.group(1)
                ground_truth = json.loads(gt_text)
            except Exception:
                continue
            
            # Extract prompts
            system_match = re.search(
                r'<\|im_start\|>system\n(.*?)<\|im_end\|>',
                text, re.DOTALL
            )
            user_match = re.search(
                r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
                text, re.DOTALL
            )
            
            if not system_match or not user_match:
                continue
            
            system_prompt = system_match.group(1)
            user_prompt = user_match.group(1)
            
            # Generate prediction
            response = model_loader.generate(system_prompt, user_prompt)
            if not response:
                results['total_samples'] += 1
                continue
            
            # Parse prediction
            try:
                pred_match = re.search(r'\[.*\]', response, re.DOTALL)
                if pred_match:
                    predictions = json.loads(pred_match.group(0))
                    results['parse_success'] += 1
                    
                    # Calculate metrics
                    for gt_item in ground_truth:
                        gt_date = gt_item['date']
                        gt_val = gt_item['forecast']
                        
                        pred_item = next(
                            (p for p in predictions if p.get('date') == gt_date),
                            None
                        )
                        
                        if pred_item and gt_val > 0:
                            pred_val = pred_item['forecast']
                            mae = abs(pred_val - gt_val)
                            mape = mae / gt_val * 100
                            results['mae_values'].append(mae)
                            results['mape_values'].append(mape)
            except Exception:
                pass
            
            results['total_samples'] += 1
        
        # Summary
        if results['mape_values']:
            results['avg_mape'] = round(np.mean(results['mape_values']), 2)
            results['median_mape'] = round(np.median(results['mape_values']), 2)
            results['avg_mae'] = round(np.mean(results['mae_values']), 2)
        
        results['parse_rate'] = (
            round(results['parse_success'] / results['total_samples'] * 100, 1)
            if results['total_samples'] > 0 else 0
        )
        
        return results


# ==========================================
# CLI ENTRY POINT
# ==========================================

def main():
    """
    CLI entry point cho fine-tuning thủ công.
    
    Usage:
        python -m forecast_system.agents.llm_finetuner [command]
        
    Commands:
        check      - Check requirements
        generate   - Generate training data only
        train      - Generate data + train
        validate   - Validate existing adapter
        status     - Show adapter status
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LLM Fine-Tuner for Forecast System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m forecast_system.agents.llm_finetuner check
  python -m forecast_system.agents.llm_finetuner generate --lookback 90
  python -m forecast_system.agents.llm_finetuner train --epochs 5
  python -m forecast_system.agents.llm_finetuner status
        """
    )
    
    parser.add_argument(
        'command',
        choices=['check', 'generate', 'train', 'validate', 'status'],
        help='Command to run'
    )
    parser.add_argument('--lookback', type=int, default=60, help='Lookback days for training data')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--max-pairs', type=int, default=5000, help='Max training pairs')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 LLM FINE-TUNER - Forecast System")
    print("=" * 60)
    
    if args.command == 'check':
        print("\n📋 Checking requirements...")
        finetuner = LLMFineTuner()
        result = finetuner.check_requirements()
        
        print(f"\n  Ready: {'✅ Yes' if result['ready'] else '❌ No'}")
        print(f"  Hardware: {result['hardware']['gpu_name']} ({result['hardware']['gpu_memory_gb']}GB)")
        
        if result['missing']:
            print(f"\n  ❌ Missing:")
            for m in result['missing']:
                print(f"     - {m}")
        
        if result['warnings']:
            print(f"\n  ⚠️ Warnings:")
            for w in result['warnings']:
                print(f"     - {w}")
        
        print(f"\n  Dependencies:")
        print(f"     torch: {'✅' if HAS_TORCH else '❌'}")
        print(f"     transformers: {'✅' if HAS_TRANSFORMERS else '❌'}")
        print(f"     peft: {'✅' if HAS_PEFT else '❌'}")
        print(f"     trl: {'✅' if HAS_TRL else '❌'}")
        print(f"     datasets: {'✅' if HAS_DATASETS else '❌'}")
        print(f"     bitsandbytes: {'✅' if HAS_BNB else '❌'}")
    
    elif args.command == 'generate':
        print(f"\n📊 Generating training data (lookback={args.lookback}d, max={args.max_pairs})...")
        
        from forecast_system.agents.master_file_agent import MasterFileAgent
        df_master = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        
        if df_master.empty:
            print("❌ No master file data found")
            return
        
        training_data = TrainingDataGenerator.generate_from_master_file(
            df_master=df_master,
            lookback_days=args.lookback,
            max_pairs=args.max_pairs,
        )
        
        if training_data:
            filepath = TrainingDataGenerator.save_training_data(training_data)
            print(f"\n✅ Generated {len(training_data)} training pairs")
            print(f"   Saved to: {filepath}")
            
            # Show sample
            print(f"\n📄 Sample training example:")
            print("-" * 40)
            sample_text = training_data[0]['text']
            if len(sample_text) > 500:
                print(sample_text[:500] + "...")
            else:
                print(sample_text)
        else:
            print("❌ Could not generate training data")
    
    elif args.command == 'train':
        print(f"\n🎯 Starting fine-tuning (epochs={args.epochs}, lora_r={args.lora_r})...")
        
        from forecast_system.agents.master_file_agent import MasterFileAgent
        df_master = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        
        if df_master.empty:
            print("❌ No master file data found")
            return
        
        # Generate training data
        training_data = TrainingDataGenerator.generate_from_master_file(
            df_master=df_master,
            lookback_days=args.lookback,
            max_pairs=args.max_pairs,
        )
        
        if not training_data:
            print("❌ Not enough training data")
            return
        
        TrainingDataGenerator.save_training_data(training_data)
        
        # Fine-tune
        config = {
            'num_epochs': args.epochs,
            'lora_r': args.lora_r,
            'batch_size': args.batch_size,
        }
        
        finetuner = LLMFineTuner(config=config)
        result = finetuner.finetune(training_data)
        
        print(f"\n{'='*60}")
        print(f"Result: {result.get('status', 'UNKNOWN')}")
        if result.get('status') == 'SUCCESS':
            print(f"  Training loss: {result.get('training_loss', 'N/A')}")
            print(f"  Duration: {result.get('elapsed_seconds', 0)/60:.1f} min")
            print(f"  Adapter: {result.get('adapter_path', 'N/A')}")
    
    elif args.command == 'validate':
        print("\n🔍 Validating fine-tuned model...")
        
        loader = FineTunedModelLoader.get_instance()
        if not loader.load():
            print("❌ No fine-tuned model found")
            return
        
        # Load some eval data
        from forecast_system.agents.master_file_agent import MasterFileAgent
        df_master = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        
        training_data = TrainingDataGenerator.generate_from_master_file(
            df_master=df_master, lookback_days=30, max_pairs=50,
        )
        
        if not training_data:
            print("❌ No data for validation")
            return
        
        # Use last 20% as eval
        eval_data = training_data[int(len(training_data) * 0.8):]
        
        result = FineTuneValidator.validate_predictions(
            loader, eval_data, max_samples=10
        )
        
        print(f"\n📊 Validation Results:")
        print(f"   Samples: {result.get('total_samples', 0)}")
        print(f"   Parse rate: {result.get('parse_rate', 0)}%")
        print(f"   Avg MAPE: {result.get('avg_mape', 'N/A')}%")
        print(f"   Median MAPE: {result.get('median_mape', 'N/A')}%")
        print(f"   Avg MAE: {result.get('avg_mae', 'N/A')} guests")
    
    elif args.command == 'status':
        print("\n📋 Adapter Status:")
        
        adapter_path = LLMFineTuner.get_latest_adapter()
        if adapter_path:
            meta_file = os.path.join(adapter_path, 'finetune_metadata.json')
            if os.path.exists(meta_file):
                with open(meta_file, encoding='utf-8') as f:
                    meta = json.load(f)
                
                print(f"   ✅ Adapter found: {adapter_path}")
                print(f"   Base model: {meta.get('base_model', 'N/A')}")
                print(f"   Trained: {meta.get('timestamp', 'N/A')}")
                print(f"   Training loss: {meta.get('training_loss', 'N/A')}")
                print(f"   Examples: {meta.get('training_examples', 'N/A')}")
                print(f"   Epochs: {meta.get('epochs', 'N/A')}")
                print(f"   LoRA rank: {meta.get('lora_r', 'N/A')}")
                print(f"   Duration: {meta.get('elapsed_seconds', 0)/60:.1f} min")
                
                needs = LLMFineTuner.should_retrain(adapter_path)
                print(f"\n   Needs retrain: {'⚠️ Yes' if needs else '✅ No'}")
            else:
                print(f"   ⚠️ Adapter found but no metadata: {adapter_path}")
        else:
            print("   ❌ No adapter found")
            print("   Run: python -m forecast_system.agents.llm_finetuner train")


if __name__ == '__main__':
    main()
