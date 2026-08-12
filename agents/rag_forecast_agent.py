"""
==============================================
RAG FORECAST AGENT - Self-Learning LLM
==============================================
Thay thế AIForecastAgent (LM Studio) bằng:
    1. Local LLM (llama-cpp-python) → không cần server
    2. RAG (ChromaDB) → nhớ kinh nghiệm qua context
    3. In-Context Learning → nhét brain memory vào prompt

Self-Learning Loop:
    predict → compare vs actual → store feedback → improve
    
Architecture:
    ┌─────────────────────────────────────────┐
    │          RAGForecastAgent                │
    │                                         │
    │   1. Retrieve context (KnowledgeStore)  │
    │   2. Build enriched prompt              │
    │   3. Local LLM inference                │
    │   4. Parse response                     │
    │   5. Store prediction for learning      │
    └─────────────────────────────────────────┘
"""

import json
import os
import re
import time
import datetime
import traceback
import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from forecast_system.config.settings import (
    PROJECT_ROOT, CURRENT_DATE, RAG_LLM_CONFIG,
)
from forecast_system.agents.data_agent import DataAgent
from forecast_system.agents.knowledge_store import KnowledgeStore
from forecast_system.utils.logger import get_logger

logger = get_logger('rag_forecast_agent')

# ==========================================
# LOCAL LLM BACKEND
# ==========================================

# Try llama-cpp-python first (local, no server needed)
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# Fallback: transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# LM Studio fallback (if local LLM fails)
try:
    from openai import OpenAI as _OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Config (from centralized settings)
RAG_CONFIG = {
    # Model paths — Primary: 7B, Fallback: 1.5B
    'model_path': RAG_LLM_CONFIG.get('model_path', 
        str(PROJECT_ROOT / 'models' / 'Qwen2.5-7B-Instruct-Q4_K_M.gguf')
    ),
    'model_fallback_path': RAG_LLM_CONFIG.get('model_fallback_path',
        str(PROJECT_ROOT / 'models' / 'qwen2.5-1.5b-instruct-q4_k_m.gguf')
    ),
    'model_type': RAG_LLM_CONFIG.get('model_type', 'gguf'),
    
    # Generation params (tuned for 7B)
    'max_tokens': RAG_LLM_CONFIG.get('max_tokens', 1200),
    'temperature': RAG_LLM_CONFIG.get('temperature', 0.1),
    'top_p': 0.9,
    'n_ctx': RAG_LLM_CONFIG.get('n_ctx', 8192),       # 8K context for 7B
    'n_gpu_layers': RAG_LLM_CONFIG.get('n_gpu_layers', -1),
    'n_threads': 4,
    
    # RAG params
    'rag_top_k': RAG_LLM_CONFIG.get('rag_top_k', 8),
    'max_retries': RAG_LLM_CONFIG.get('max_retries', 2),
}


# ==========================================
# LOCAL LLM WRAPPER
# ==========================================

class LocalLLM:
    """
    Wrapper cho local LLM inference.
    
    Supports:
    - Fine-tuned model (LoRA adapter, highest priority)
    - llama-cpp-python (GGUF format, Metal acceleration on Mac)
    - transformers (HuggingFace models)
    """
    
    _instance = None
    
    def __init__(self):
        self._model = None
        self._backend = None
        self._model_name = None
        self._finetuned_loader = None
    
    @classmethod
    def get_instance(cls) -> 'LocalLLM':
        """Singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_model(self):
        """Load LLM model with fallback chain: Fine-tuned → 7B → 1.5B → LM Studio."""
        if self._model is not None:
            return
        
        # ── Priority 0: Fine-tuned model (LoRA adapter) ──
        try:
            from forecast_system.agents.llm_finetuner import FineTunedModelLoader
            ft_loader = FineTunedModelLoader.get_instance()
            if ft_loader.load():
                self._finetuned_loader = ft_loader
                self._model = 'finetuned'  # Sentinel value
                self._backend = 'finetuned'
                self._model_name = 'Forecast Fine-tuned (LoRA)'
                logger.info("✅ Using fine-tuned model for inference")
                return
        except Exception as e:
            logger.debug(f"Fine-tuned model not available: {e}")
        
        model_path = RAG_CONFIG['model_path']
        model_fallback = RAG_CONFIG.get('model_fallback_path', '')
        model_type = RAG_CONFIG['model_type']
        
        if model_type == 'gguf' and HAS_LLAMA_CPP:
            # Try primary model (7B)
            if os.path.exists(model_path):
                try:
                    self._load_gguf(model_path)
                    return
                except Exception as e:
                    logger.warning(f"Primary model failed: {e}")
            else:
                logger.warning(f"Primary model not found: {model_path}")
            
            # Try fallback model (1.5B)
            if model_fallback and os.path.exists(model_fallback):
                logger.info(f"Trying fallback model: {model_fallback}")
                try:
                    self._load_gguf(model_fallback)
                    return
                except Exception as e:
                    logger.warning(f"Fallback model failed: {e}")
        
        # LM Studio fallback — DISABLED on Windows (causes timeout delays)
        # To re-enable, start LM Studio server first, then set env:
        #   USE_LM_STUDIO_FALLBACK=true
        use_lm_studio = os.getenv('USE_LM_STUDIO_FALLBACK', 'false').lower() in ('true', '1')
        if use_lm_studio and HAS_OPENAI:
            logger.info("Falling back to LM Studio server...")
            try:
                self._load_lm_studio()
                return
            except Exception as e:
                logger.warning(f"LM Studio fallback failed: {e}")
        elif not use_lm_studio:
            logger.info("LM Studio fallback disabled (set USE_LM_STUDIO_FALLBACK=true to enable)")
        
        if HAS_TRANSFORMERS:
            self._load_transformers(model_path)
        else:
            raise RuntimeError(
                "No LLM backend available. Options:\n"
                "  1. Fine-tune a model (see llm_finetuner.py)\n"
                "  2. Download GGUF model to models/ directory\n"
                "  3. pip install llama-cpp-python\n"
                "  4. pip install transformers torch"
            )
    
    def _load_gguf(self, model_path: str):
        """Load GGUF model via llama-cpp-python."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_path}")
        
        model_size_gb = os.path.getsize(model_path) / (1024**3)
        logger.info(f"Loading GGUF model: {os.path.basename(model_path)} ({model_size_gb:.1f}GB)")
        self._model = Llama(
            model_path=model_path,
            n_ctx=RAG_CONFIG['n_ctx'],
            n_gpu_layers=RAG_CONFIG['n_gpu_layers'],
            n_threads=RAG_CONFIG['n_threads'],
            verbose=False,
        )
        self._backend = 'llama_cpp'
        self._model_name = os.path.basename(model_path)
        logger.info(f"✅ GGUF model loaded: {self._model_name} (Metal acceleration)")
    
    def _load_lm_studio(self):
        """Load LM Studio as fallback backend."""
        from forecast_system.config.settings import LM_STUDIO_CONFIG
        client = _OpenAI(
            base_url=LM_STUDIO_CONFIG['base_url'],
            api_key=LM_STUDIO_CONFIG['api_key'],
            timeout=float(LM_STUDIO_CONFIG['timeout']),
        )
        # Quick health check
        client.models.list()
        self._model = client
        self._backend = 'lm_studio'
        self._model_name = LM_STUDIO_CONFIG['model']
        logger.info(f"✅ LM Studio connected: {self._model_name}")
    
    def _load_transformers(self, model_name: str):
        """
        Load model via HuggingFace transformers.
        
        Supports:
        - GGUF files: loads via transformers GGUF support (requires transformers>=4.35)
        - HuggingFace model names: loads normally from HF Hub
        """
        logger.info(f"Loading transformers model: {model_name}")
        
        # Detect if model_name is a GGUF file path
        is_gguf = model_name.lower().endswith('.gguf') and os.path.exists(model_name)
        
        if is_gguf:
            # For GGUF files, use HuggingFace model name to load tokenizer/config
            # and pass gguf_file parameter for model weights
            gguf_path = model_name
            gguf_filename = os.path.basename(gguf_path)
            gguf_dir = os.path.dirname(gguf_path)
            
            # Determine the HF model name from the GGUF filename
            # e.g., Qwen2.5-7B-Instruct-Q4_K_M.gguf → Qwen/Qwen2.5-7B-Instruct
            #       qwen2.5-1.5b-instruct-q4_k_m.gguf → Qwen/Qwen2.5-1.5B-Instruct
            hf_model_name = self._resolve_hf_name_from_gguf(gguf_filename)
            
            logger.info(f"Loading GGUF via transformers: {gguf_filename}")
            logger.info(f"  HF base model for tokenizer: {hf_model_name}")
            
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            try:
                # Load tokenizer from HuggingFace Hub
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name, trust_remote_code=True
                )
                # Load GGUF: pass directory as pretrained path, filename as gguf_file
                model = AutoModelForCausalLM.from_pretrained(
                    gguf_dir,
                    gguf_file=gguf_filename,
                    dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"GGUF loading via transformers failed: {e}")
                logger.info(f"Trying direct HF model download: {hf_model_name}")
                # Fallback: load HF model directly (downloads from hub)
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            
            self._model_name = gguf_filename
        else:
            # Standard HuggingFace model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model_name = model_name
        
        # Store tokenizer separately for chat template usage
        self._tokenizer = tokenizer
        self._hf_model = model
        self._model = 'transformers_loaded'  # Sentinel
        self._backend = 'transformers'
        logger.info(f"✅ Transformers model loaded: {self._model_name}")
    
    @staticmethod
    def _resolve_hf_name_from_gguf(gguf_filename: str) -> str:
        """
        Resolve HuggingFace model name from GGUF filename.
        
        Examples:
            Qwen2.5-7B-Instruct-Q4_K_M.gguf → Qwen/Qwen2.5-7B-Instruct
            qwen2.5-1.5b-instruct-q4_k_m.gguf → Qwen/Qwen2.5-1.5B-Instruct
        """
        import re
        name = gguf_filename.replace('.gguf', '')
        # Remove quantization suffix (Q4_K_M, Q5_K_S, etc.)
        name = re.sub(r'[-_]Q\d+[_\w]*$', '', name, flags=re.IGNORECASE)
        
        # Known model mappings
        mappings = {
            'qwen2.5-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen2.5-3b-instruct': 'Qwen/Qwen2.5-3B-Instruct',
            'qwen2.5-1.5b-instruct': 'Qwen/Qwen2.5-1.5B-Instruct',
            'qwen2.5-0.5b-instruct': 'Qwen/Qwen2.5-0.5B-Instruct',
            'qwen2.5-14b-instruct': 'Qwen/Qwen2.5-14B-Instruct',
            'qwen2.5-32b-instruct': 'Qwen/Qwen2.5-32B-Instruct',
            'qwen2.5-72b-instruct': 'Qwen/Qwen2.5-72B-Instruct',
        }
        
        name_lower = name.lower()
        if name_lower in mappings:
            return mappings[name_lower]
        
        # Generic: assume Qwen model
        if 'qwen' in name_lower:
            return f"Qwen/{name}"
        
        # Default fallback
        return name
    
    def generate(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Generate response from LLM (fine-tuned / GGUF / LM Studio / transformers).
        
        Returns:
            str response or None if failed
        """
        if self._model is None:
            self.load_model()
        
        try:
            if self._backend == 'finetuned':
                return self._generate_finetuned(system_prompt, user_prompt)
            elif self._backend == 'llama_cpp':
                return self._generate_llama_cpp(system_prompt, user_prompt)
            elif self._backend == 'lm_studio':
                return self._generate_lm_studio(system_prompt, user_prompt)
            elif self._backend == 'transformers':
                return self._generate_transformers(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"LLM generation failed ({self._backend}): {e}")
            traceback.print_exc()
            return None
    
    def _generate_finetuned(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate with fine-tuned model (LoRA adapter)."""
        if self._finetuned_loader and self._finetuned_loader.is_loaded:
            return self._finetuned_loader.generate(
                system_prompt, user_prompt,
                max_tokens=RAG_CONFIG['max_tokens'],
                temperature=RAG_CONFIG['temperature'],
            )
        return None
    
    def _generate_llama_cpp(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate with llama-cpp-python."""
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=RAG_CONFIG['max_tokens'],
            temperature=RAG_CONFIG['temperature'],
            top_p=RAG_CONFIG['top_p'],
        )
        
        content = response['choices'][0]['message']['content']
        return content if content else None
    
    def _generate_lm_studio(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate via LM Studio (OpenAI-compatible API)."""
        from forecast_system.config.settings import LM_STUDIO_CONFIG
        response = self._model.chat.completions.create(
            model=LM_STUDIO_CONFIG['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=RAG_CONFIG['max_tokens'],
            temperature=RAG_CONFIG['temperature'],
        )
        content = response.choices[0].message.content
        return content if content else None
    
    def _generate_transformers(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Generate with HuggingFace transformers.
        Uses apply_chat_template for proper Qwen chat formatting.
        Optimized for GPU inference with inference_mode and KV-cache.
        """
        import torch
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Determine device (handle offloaded models)
        try:
            device = next(self._hf_model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Resolve pad_token_id to suppress generation warnings
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        
        # Use inference_mode for faster GPU execution (no grad tracking)
        with torch.inference_mode():
            # Use apply_chat_template if available (Qwen, Llama3, etc.)
            if hasattr(self._tokenizer, 'apply_chat_template'):
                tokenized = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                
                # Handle both dict-like (BatchEncoding) and plain tensor returns
                if isinstance(tokenized, dict) or hasattr(tokenized, 'input_ids'):
                    input_ids = tokenized['input_ids'].to(device)
                    attention_mask = tokenized.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                else:
                    # Plain tensor
                    input_ids = tokenized.to(device)
                    attention_mask = None
                
                input_len = input_ids.shape[1]
                
                gen_kwargs = {
                    'max_new_tokens': RAG_CONFIG['max_tokens'],
                    'temperature': max(RAG_CONFIG['temperature'], 0.01),
                    'top_p': RAG_CONFIG['top_p'],
                    'do_sample': True,
                    'use_cache': True,       # KV-cache for faster autoregressive gen
                    'pad_token_id': pad_token_id,
                }
                if attention_mask is not None:
                    gen_kwargs['attention_mask'] = attention_mask
                
                outputs = self._hf_model.generate(input_ids, **gen_kwargs)
                
                # Decode only the generated tokens (exclude input)
                generated_ids = outputs[0][input_len:]
                text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                # Fallback: basic prompt formatting
                prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
                inputs = self._tokenizer(prompt, return_tensors="pt")
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                input_len = input_ids.shape[1]
                
                gen_kwargs = {
                    'max_new_tokens': RAG_CONFIG['max_tokens'],
                    'temperature': max(RAG_CONFIG['temperature'], 0.01),
                    'top_p': RAG_CONFIG['top_p'],
                    'do_sample': True,
                    'use_cache': True,
                    'pad_token_id': pad_token_id,
                }
                if attention_mask is not None:
                    gen_kwargs['attention_mask'] = attention_mask
                
                outputs = self._hf_model.generate(input_ids, **gen_kwargs)
                
                generated_ids = outputs[0][input_len:]
                text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return text.strip() if text and text.strip() else None
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def backend_name(self) -> str:
        name = self._backend or 'none'
        if self._model_name:
            name += f' ({self._model_name})'
        return name


# ==========================================
# RAG FORECAST AGENT
# ==========================================

class RAGForecastAgent:
    """
    AI Forecast Agent with RAG + In-Context Learning.
    
    Thay thế AIForecastAgent (LM Studio).
    
    Key differences:
    - Local LLM (no server needed)
    - RAG context từ KnowledgeStore
    - In-Context Learning từ brain memory
    - Self-learning loop
    
    Usage:
        agent = RAGForecastAgent()
        
        # Index knowledge (chạy 1 lần hoặc mỗi pipeline run)
        agent.update_knowledge(brain_memory, df_master, df_train)
        
        # Generate forecast
        response = agent.generate_forecast(res_code, history_text, next_days, report)
        parsed = agent.parse_response(response)
    """
    
    def __init__(self, knowledge_store: KnowledgeStore = None):
        self._store = knowledge_store or KnowledgeStore()
        self._llm = LocalLLM.get_instance()
    
    # ==========================================
    # KNOWLEDGE INDEXING
    # ==========================================
    
    def update_knowledge(
        self,
        brain_memory: Dict = None,
        df_master: pd.DataFrame = None,
        df_train: pd.DataFrame = None,
    ):
        """
        Update RAG knowledge base.
        Gọi mỗi lần chạy pipeline để cập nhật kinh nghiệm.
        """
        
        if brain_memory:
            logger.info("📚 Indexing brain memory → Knowledge Store")
            self._store.index_brain_memory(brain_memory)
        
        if df_master is not None and not df_master.empty:
            logger.info("📚 Indexing prediction feedback → Knowledge Store")
            self._store.index_prediction_feedback(df_master)
        
        if df_train is not None and not df_train.empty:
            logger.info("📚 Indexing restaurant patterns → Knowledge Store")
            self._store.index_all_restaurant_patterns(df_train)
        
        stats = self._store.get_stats()
        logger.info(f"📊 Knowledge Store: {stats}")
    
    # ==========================================
    # PROMPT PREPARATION (backward compatible)
    # ==========================================
    
    @staticmethod
    def prepare_prompt_data(df_res, vn_holidays):
        """Backward compatible: basic 21-day history."""
        daily = df_res.groupby('date')['guest_count'].sum().reset_index().sort_values('date')
        recent_21 = daily.tail(21).copy()
        
        history_text = "HISTORY (Date | Weekday | Guest Count):\n"
        for _, row in recent_21.iterrows():
            d = row['date']
            wd = d.strftime('%A')
            cnt = int(row['guest_count'])
            is_hol = "HOLIDAY" if d in vn_holidays else ""
            history_text += f"- {d} ({wd}): {cnt} {is_hol}\n"
        return history_text
    
    @staticmethod
    def prepare_weighted_prompt_data(df_res, target_date, vn_holidays):
        """Backward compatible: weighted 30/60/90 day stats."""
        history_text = "WEIGHTED ANALYSIS:\n\n"
        
        history_text += "📊 RECENT 30 DAYS (70% importance):\n"
        for wd_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']:
            avg_30 = DataAgent.get_window_statistics(df_res, 30, wd_name, target_date)
            if avg_30:
                history_text += f"  {wd_name}: ~{int(avg_30)} guests\n"
        
        history_text += "\n📈 60-DAY TREND (10% importance):\n"
        for wd_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']:
            avg_60 = DataAgent.get_window_statistics(df_res, 60, wd_name, target_date)
            if avg_60:
                history_text += f"  {wd_name}: ~{int(avg_60)} guests\n"
        
        history_text += "\n🗓️ 90-DAY SEASONAL (10% importance):\n"
        for wd_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']:
            avg_90 = DataAgent.get_window_statistics(df_res, 90, wd_name, target_date)
            if avg_90:
                history_text += f"  {wd_name}: ~{int(avg_90)} guests\n"
        
        daily = df_res.groupby('date')['guest_count'].sum().reset_index().sort_values('date')
        recent_7 = daily.tail(7)
        history_text += "\n📅 LAST 7 DAYS:\n"
        for _, row in recent_7.iterrows():
            d = row['date']
            wd = d.strftime('%a')
            cnt = int(row['guest_count'])
            history_text += f"  {d} ({wd}): {cnt}\n"
        
        return history_text
    
    @staticmethod
    def prepare_enhanced_prompt(df_res, target_date, vn_holidays, analysis_report=None):
        """Backward compatible: enhanced prompt with analysis."""
        prompt = RAGForecastAgent.prepare_weighted_prompt_data(
            df_res, target_date, vn_holidays
        )
        
        if analysis_report:
            from forecast_system.agents.analysis_agent import AnalysisAgent
            report_text = AnalysisAgent.format_report_for_prompt(analysis_report)
            prompt = report_text + "\n\n" + prompt
        
        return prompt
    
    # ==========================================
    # FORECAST GENERATION (RAG-Enhanced)
    # ==========================================
    
    def generate_forecast(
        self,
        res_code: str,
        history_text: str,
        next_days_info: List[Dict],
        analysis_report: Dict = None,
        brain_memory: Dict = None,
    ) -> Optional[str]:
        """
        Generate forecast with RAG context + In-Context Learning.
        
        Replaces AIForecastAgent.generate_forecast()
        """
        max_retries = RAG_CONFIG['max_retries']
        
        # 1. Build future targets text
        future_text = "FORECAST TARGETS (Date | Weekday | Event):\n"
        first_day_info = None
        for d in next_days_info:
            if first_day_info is None:
                first_day_info = d
            evt = []
            h_type = d.get('holiday_type')
            if h_type:
                evt.append(f"HOLIDAY:{h_type}")
            elif d.get('is_pre_holiday'):
                evt.append(f"PRE_HOLIDAY:{d.get('pre_post_type', 'PRE_HOLIDAY')}")
            elif d.get('is_post_holiday'):
                evt.append(f"POST_HOLIDAY:{d.get('pre_post_type', 'POST_HOLIDAY')}")
            elif d['is_holiday']:
                evt.append("HOLIDAY")
            if d.get('is_veg', False):
                evt.append("VEG_DAY")
            if d['weekday'] in ['Saturday', 'Sunday']:
                evt.append("WEEKEND")
            if d.get('closed_likely'):
                evt.append("⚠️LIKELY_CLOSED")
            evt_str = " | ".join(evt) if evt else "Normal"
            future_text += f"- {d['date']} ({d['weekday']}) : {evt_str}\n"
        
        # 2. RAG: Retrieve relevant context
        rag_context = ""
        try:
            target_date = first_day_info['date'] if first_day_info else CURRENT_DATE
            target_weekday = first_day_info.get('weekday', '') if first_day_info else ''
            is_holiday = first_day_info.get('is_holiday', False) if first_day_info else False
            
            rag_context = self._store.retrieve_context(
                res_code=res_code,
                target_date=target_date,
                weekday=target_weekday,
                is_holiday=is_holiday,
                top_k=RAG_CONFIG['rag_top_k'],
            )
        except Exception as e:
            logger.debug(f"RAG retrieval skipped: {e}")
        
        # 3. In-Context Learning: Brain memory injection
        brain_context = ""
        if brain_memory:
            brain_context = self._format_brain_for_prompt(brain_memory, res_code)
        
        # 4. Build prompts
        system_prompt = self._build_system_prompt(analysis_report)
        
        user_prompt = f"Restaurant: {res_code}\n\n"
        
        # RAG context (retrieved experience)
        if rag_context:
            user_prompt += (
                "🧠 LEARNED KNOWLEDGE (from past predictions and errors):\n"
                f"{rag_context}\n\n"
            )
        
        # Brain context is descriptive only. Numeric corrections still happen
        # later in the ensemble pipeline, so do not instruct the LLM to apply
        # hard bias offsets here.
        if brain_context:
            user_prompt += (
                "📝 LEARNED MODEL BEHAVIOR (reference only):\n"
                f"{brain_context}\n\n"
            )
        
        user_prompt += f"{history_text}\n\n{future_text}\n\nGenerate JSON forecast:"
        
        # 5. LLM Generation with retry
        for attempt in range(1, max_retries + 1):
            try:
                result = self._llm.generate(system_prompt, user_prompt)
                
                if result:
                    logger.info(
                        f"RAG forecast generated for {res_code} "
                        f"(attempt {attempt}, backend: {self._llm.backend_name})"
                    )
                    return result
                else:
                    logger.warning(f"Empty response for {res_code} (attempt {attempt})")
                    
            except Exception as e:
                logger.warning(
                    f"LLM call failed for {res_code} "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(attempt * 2)
        
        logger.error(f"All {max_retries} LLM attempts failed for {res_code}")
        return None
    
    # ==========================================
    # BRAIN → PROMPT (In-Context Learning)
    # ==========================================
    
    @staticmethod
    def _format_brain_for_prompt(brain_memory: Dict, res_code: str) -> str:
        """
        Format brain memory thành context text cho prompt.
        
        Đây là In-Context Learning - cho LLM biết:
        - Model thường sai ở đâu
        - Cần điều chỉnh bao nhiêu
        - Strategy nào tốt nhất
        """
        parts = []
        
        # Global patterns
        global_p = brain_memory.get('global_patterns', {})
        if global_p:
            holiday_bias = global_p.get('holiday_bias_pct', 0)
            weekend_bias = global_p.get('weekend_bias', 0)
            if holiday_bias:
                direction = "over" if holiday_bias > 0 else "under"
                parts.append(
                    f"⚠ SYSTEM holiday bias: tends to {direction}-predict by "
                    f"{abs(holiday_bias):.0f}%."
                )
            if weekend_bias:
                direction = "over" if weekend_bias > 0 else "under"
                parts.append(
                    f"⚠ SYSTEM weekend bias: tends to {direction}-predict by "
                    f"{abs(weekend_bias):.0f} guests."
                )
        
        # Restaurant-specific
        res_mem = brain_memory.get('restaurants', {}).get(str(res_code), {})
        if res_mem:
            overall_bias = res_mem.get('overall_bias', 0)
            if abs(overall_bias) > 2:
                direction = "over" if overall_bias > 0 else "under"
                parts.append(
                    f"🎯 This restaurant raw AI forecast tends to {direction}-predict "
                    f"by ~{abs(overall_bias):.0f} guests on average."
                )
            
            # Weekday biases
            wd_bias = res_mem.get('weekday_bias', {})
            if wd_bias:
                significant = {wd: b for wd, b in wd_bias.items() if abs(b) > 3}
                if significant:
                    bias_text = ", ".join(
                        f"{wd}: {'over' if b > 0 else 'under'} by {abs(b):.0f}"
                        for wd, b in significant.items()
                    )
                    parts.append(f"📊 Weekday biases: {bias_text}")
            
            # Best strategy
            best = res_mem.get('best_strategy', '')
            ml_mape = res_mem.get('ml_mape', 'N/A')
            ai_mape = res_mem.get('ai_mape', 'N/A')
            if best:
                parts.append(
                    f"📈 Best strategy: {best} (ML MAPE: {ml_mape}%, AI MAPE: {ai_mape}%)"
                )
            
            # MAPE trend
            mape_hist = res_mem.get('mape_history', [])
            if len(mape_hist) >= 2:
                trend = "IMPROVING ✅" if mape_hist[-1] < mape_hist[-2] else "DECLINING ❌"
                parts.append(f"📉 Accuracy trend: {trend} (MAPE: {mape_hist[-3:]})")
            
            # Recent issues
            issues = res_mem.get('issues', [])
            if issues:
                recent_issues = issues[-3:]
                for iss in recent_issues:
                    parts.append(
                        f"⚠ Past issue on {iss.get('date')}: "
                        f"{iss.get('type', 'UNKNOWN')} "
                        f"(error: {iss.get('error_pct', 0):.0f}%)"
                    )
        
        return "\n".join(parts) if parts else ""
    
    # ==========================================
    # SYSTEM PROMPT
    # ==========================================
    
    @staticmethod
    def _build_system_prompt(analysis_report: Dict = None) -> str:
        """Build system prompt with RAG + In-Context Learning guidance."""
        
        base_prompt = """
You are a Self-Learning Demand Planner for a restaurant chain in Vietnam.
You have MEMORY of past predictions and their errors (shown in LEARNED KNOWLEDGE section).
You have LEARNED MODEL BEHAVIOR notes that describe recurring bias patterns.

CRITICAL: Use the LEARNED KNOWLEDGE and LEARNED MODEL BEHAVIOR as context,
but do not mechanically apply fixed offsets. Produce the best raw forecast from history,
trend, weekday pattern, and event context.

LOGIC:
1. Start with WEEKLY PATTERN from History (each weekday has its own baseline).
2. Cross-reference with LEARNED KNOWLEDGE (past similar predictions and their errors).
3. Use LEARNED MODEL BEHAVIOR to avoid repeating obvious mistakes.
4. Apply TREND from Analysis Report (Growth/Decline).
5. EVENTS impact (VERY IMPORTANT):
   - HOLIDAY:TET_NGUYEN_DAN: -80% to -100% (most CLOSED during Lunar New Year!)
   - HOLIDAY:NATIONAL_DAY: +10% to +30% (people eat out)
   - HOLIDAY:HUNG_KINGS: +5% to +20%
   - HOLIDAY:LIBERATION_DAY: +10% to +25% (30/4)
   - HOLIDAY:LABOR_DAY: +10% to +25% (1/5)
   - HOLIDAY:TET_DUONG_LICH: +5% to +15%
   - PRE_HOLIDAY:PRE_TET: +20% to +30%
   - POST_HOLIDAY:POST_TET: -20% to -40%
   - VEG_DAY: -5% to -15%
   - WEEKEND: Use weekend pattern (usually higher)
   - ⚠️LIKELY_CLOSED: Forecast ZERO!
6. Round to nearest integer. Never forecast negative.
"""
        
        if analysis_report:
            trend = analysis_report.get('trend', 'STABLE')
            score = analysis_report.get('trend_score', 0)
            category = analysis_report.get('category', 'STANDARD')
            
            base_prompt += f"""
RESTAURANT CONTEXT:
- Category: {category}
- Trend: {trend} (score: {score})
- Confidence: {analysis_report.get('confidence', 'N/A')}
- Apply trend adjustment: {"increase" if score > 0 else "decrease" if score < 0 else "maintain"} baseline by ~{abs(score) / 10:.1f}% per week
"""
        
        base_prompt += """
OUTPUT JSON ONLY (no explanation):
[
    {"date": "YYYY-MM-DD", "forecast": 150},
    ...
]
"""
        return base_prompt
    
    # ==========================================
    # RESPONSE PARSING (same as AIForecastAgent)
    # ==========================================
    
    @staticmethod
    def parse_response(text: str) -> Optional[List[Dict]]:
        """Parse JSON array from LLM response."""
        if not text:
            return None
        
        try:
            data = json.loads(text.strip())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, ValueError):
            pass
        
        try:
            objects = re.findall(r'\{[^{}]+\}', text)
            if objects:
                parsed = []
                for obj_str in objects:
                    try:
                        obj = json.loads(obj_str)
                        if 'date' in obj and 'forecast' in obj:
                            parsed.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        continue
                if parsed:
                    return parsed
        except Exception:
            pass
        
        logger.warning(f"Failed to parse LLM response: {text[:200]}...")
        return None
    
    # ==========================================
    # SELF-LEARNING LOOP
    # ==========================================
    
    def learn_from_actuals(
        self,
        df_master: pd.DataFrame,
        strategy_map: Dict[str, str] = None,  # type: ignore[reportArgumentType]
    ):
        """
        Self-learning: Store prediction vs actual feedback.
        
        Called after actuals become available (during next pipeline run).
        Uses AI_Raw_Daily_Forecast (when available) for cleaner learning signal.
        """
        ai_available = (
            df_master['AI_Forecast_Available'].fillna(False).astype(bool)
            if 'AI_Forecast_Available' in df_master.columns
            else True
        )
        mask = (
            pd.notna(df_master.get('AI_Raw_Daily_Forecast')) &  # type: ignore[reportOperatorIssue]
            pd.notna(df_master.get('Actual_Guest')) &
            (df_master['Actual_Guest'] >= 0) &
            ai_available
        )
        df = df_master[mask].copy()
        
        if df.empty:
            return
        
        # Aggregate to daily
        daily = df.groupby(['Restaurant_Code', 'Date']).agg({
            'AI_Raw_Daily_Forecast': 'first',
            'Actual_Guest': 'sum',
            'Weekday': 'first',
            'Is_Holiday': 'first',
        }).reset_index()
        
        # Only learn from recent data (last 14 days)
        cutoff = CURRENT_DATE - datetime.timedelta(days=14)
        daily = daily[daily['Date'] >= cutoff]
        
        n_learned = 0
        for _, row in daily.iterrows():
            try:
                strategy = ''
                if strategy_map:
                    strategy = strategy_map.get(str(row['Restaurant_Code']), 'UNKNOWN')
                
                self._store.learn_from_results(
                    res_code=str(row['Restaurant_Code']),
                    date_obj=row['Date'],  # type: ignore[reportArgumentType]
                    predicted=float(row['AI_Raw_Daily_Forecast']),
                    actual=float(row['Actual_Guest']),
                    weekday=str(row['Weekday']),
                    strategy=strategy,
                    is_holiday=bool(row.get('Is_Holiday', False)),
                )
                n_learned += 1
            except Exception:
                continue
        
        logger.info(f"🧠 Self-learning: stored {n_learned} feedback records")
    
    # ==========================================
    # KNOWLEDGE STORE ACCESS
    # ==========================================
    
    @property
    def knowledge_stats(self) -> Dict:
        """Get knowledge store statistics."""
        return self._store.get_stats()
    
    @property
    def store(self) -> KnowledgeStore:
        """Direct access to knowledge store."""
        return self._store
