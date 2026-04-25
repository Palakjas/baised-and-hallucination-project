#!/usr/bin/env python3
"""
Automated script to download and organize all datasets
for the XAI Bias & Hallucination Detection System.

Usage:
    python setup_datasets.py
"""

import os
from pathlib import Path
from datasets import load_dataset
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create the data directory structure"""
    base_dir = Path('./data')
    subdirs = [
        'bias',
        'hallucination',
        'explainability',
        'combined',
        'processed'
    ]
    
    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Directory structure created")


def download_datasets():
    """Download all datasets"""
    
    print("\n" + "="*60)
    print("DOWNLOADING DATASETS")
    print("="*60)
    
    results = {}
    
    # Bias datasets
    print("\n1. BIAS DETECTION DATASETS")
    print("-" * 40)
    
    try:
        print("  Downloading BOLD...")
        bold = load_dataset('facebook/bold')
        results['bold'] = f"✓ {len(bold['train'])} samples"
        print(f"  ✓ BOLD downloaded ({len(bold['train'])} samples)")
    except Exception as e:
        results['bold'] = f"✗ {e}"
        print(f"  ✗ BOLD failed: {e}")
    
    try:
        print("  Downloading e-SNLI...")
        e_snli = load_dataset('e_snli')
        results['e_snli'] = f"✓ {len(e_snli['train'])} samples"
        print(f"  ✓ e-SNLI downloaded ({len(e_snli['train'])} samples)")
    except Exception as e:
        results['e_snli'] = f"✗ {e}"
        print(f"  ✗ e-SNLI failed: {e}")
    
    # Hallucination datasets
    print("\n2. HALLUCINATION DETECTION DATASETS")
    print("-" * 40)
    
    try:
        print("  Downloading TruthfulQA...")
        truthful_qa = load_dataset('truthful_qa', 'generation')
        results['truthful_qa'] = f"✓ {len(truthful_qa['validation'])} samples"
        print(f"  ✓ TruthfulQA downloaded ({len(truthful_qa['validation'])} samples)")
    except Exception as e:
        results['truthful_qa'] = f"✗ {e}"
        print(f"  ✗ TruthfulQA failed: {e}")
    
    try:
        print("  Downloading FEVER...")
        fever = load_dataset('fever', 'v1.0')
        train_size = len(fever['train']) if 'train' in fever else "N/A"
        results['fever'] = f"✓ {train_size} samples"
        print(f"  ✓ FEVER downloaded ({train_size} samples)")
    except Exception as e:
        results['fever'] = f"✗ {e}"
        print(f"  ✗ FEVER failed: {e}")
    
    try:
        print("  Downloading SQuAD...")
        squad = load_dataset('squad')
        results['squad'] = f"✓ {len(squad['train'])} samples"
        print(f"  ✓ SQuAD downloaded ({len(squad['train'])} samples)")
    except Exception as e:
        results['squad'] = f"✗ {e}"
        print(f"  ✗ SQuAD failed: {e}")
    
    try:
        print("  Downloading HotpotQA...")
        hotpot = load_dataset('hotpot_qa', 'distractor')
        results['hotpot_qa'] = f"✓ {len(hotpot['train'])} samples"
        print(f"  ✓ HotpotQA downloaded ({len(hotpot['train'])} samples)")
    except Exception as e:
        results['hotpot_qa'] = f"✗ {e}"
        print(f"  ✗ HotpotQA failed: {e}")
    
    # Save download report
    print("\n" + "="*60)
    print("DOWNLOAD REPORT")
    print("="*60)
    for name, status in results.items():
        print(f"  {name:20s} : {status}")
    
    report_path = Path('./data/download_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {report_path}")
    
    print("\n" + "="*60)
    print("✓ Dataset download complete!")
    print("="*60)
    print("\nNote: Datasets are cached in ~/.cache/huggingface/datasets/")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("  XAI Bias & Hallucination Detection - Dataset Setup")
    print("="*60)
    create_directory_structure()
    download_datasets()
    print("\n✓ Setup complete! You can now use the datasets.")
