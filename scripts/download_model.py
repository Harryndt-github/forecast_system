#!/usr/bin/env python3
"""
Download LLM Model for RAG Forecast Agent.

Usage:
    python scripts/download_model.py              # Download default 7B model
    python scripts/download_model.py --model 1.5b # Download 1.5B fallback
    python scripts/download_model.py --list        # List available models
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

MODELS_DIR = PROJECT_ROOT / 'models'

# Available models
AVAILABLE_MODELS = {
    '7b': {
        'repo': 'bartowski/Qwen2.5-7B-Instruct-GGUF',
        'filename': 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
        'size': '~4.4 GB',
        'description': 'Qwen2.5 7B Instruct (Q4_K_M) - Recommended for production',
        'min_ram': '8 GB',
    },
    '3b': {
        'repo': 'bartowski/Qwen2.5-3B-Instruct-GGUF',
        'filename': 'Qwen2.5-3B-Instruct-Q4_K_M.gguf',
        'size': '~2.0 GB',
        'description': 'Qwen2.5 3B Instruct (Q4_K_M) - Balanced speed/quality',
        'min_ram': '4 GB',
    },
    '1.5b': {
        'repo': 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
        'filename': 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
        'size': '~1.1 GB',
        'description': 'Qwen2.5 1.5B Instruct (Q4_K_M) - Fast but lower quality',
        'min_ram': '2 GB',
    },
}

DEFAULT_MODEL = '7b'


def list_models():
    """List available models."""
    print("\n📦 Available Models:")
    print("=" * 70)
    for key, info in AVAILABLE_MODELS.items():
        installed = "✅" if (MODELS_DIR / info['filename']).exists() else "  "
        default = " (DEFAULT)" if key == DEFAULT_MODEL else ""
        print(f"  {installed} {key:>5s}{default}")
        print(f"         {info['description']}")
        print(f"         Size: {info['size']}  |  Min RAM: {info['min_ram']}")
        print(f"         Repo: {info['repo']}")
        print()
    print("=" * 70)
    print(f"Usage: python {__file__} --model <key>")


def download_model(model_key: str):
    """Download a specific model."""
    if model_key not in AVAILABLE_MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"   Available: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    
    info = AVAILABLE_MODELS[model_key]
    target_path = MODELS_DIR / info['filename']
    
    if target_path.exists():
        size_gb = target_path.stat().st_size / (1024**3)
        print(f"✅ Model already exists: {target_path} ({size_gb:.1f} GB)")
        return True
    
    print(f"\n🔽 Downloading: {info['description']}")
    print(f"   From: {info['repo']}")
    print(f"   File: {info['filename']}")
    print(f"   Size: {info['size']}")
    print(f"   To:   {target_path}")
    print()
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface-hub not installed. Run: pip install huggingface-hub")
        return False
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        path = hf_hub_download(
            repo_id=info['repo'],
            filename=info['filename'],
            local_dir=str(MODELS_DIR),
        )
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"\n✅ Download complete: {path} ({size_gb:.1f} GB)")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download LLM model for RAG Forecast Agent')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help=f'Model size to download (default: {DEFAULT_MODEL})')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--all', action='store_true',
                        help='Download all models')
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if args.all:
        for key in AVAILABLE_MODELS:
            download_model(key)
        return
    
    download_model(args.model)


if __name__ == '__main__':
    main()
