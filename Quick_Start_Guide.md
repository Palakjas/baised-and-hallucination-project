# Quick Start Guide: Dataset Installation & Download

## Installation Requirements

### 1. Install Required Libraries

```bash
# Core NLP libraries
pip install datasets transformers torch

# Data processing
pip install pandas numpy scikit-learn

# Additional utilities
pip install tqdm requests

# Specific dataset tools
pip install huggingface-hub
```

### 2. Complete Installation Script

```bash
# Create virtual environment (recommended)
python -m venv nlp_env
source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate

# Install all requirements
pip install -r requirements.txt
```

**requirements.txt:**
```
datasets>=2.10.0
transformers>=4.30.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
tqdm>=4.65.0
requests>=2.28.0
huggingface-hub>=0.16.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## Dataset Download Methods

### Method 1: Using HuggingFace Datasets (Recommended)

**Most Convenient - Automatic Download & Caching**

```python
from datasets import load_dataset

# Bias Datasets
bold = load_dataset('facebook/bold')
e_snli = load_dataset('e_snli')

# Hallucination Datasets
truthful_qa = load_dataset('truthful_qa', 'generation')
fever = load_dataset('fever')
squad = load_dataset('squad')

# All datasets auto-cache to ~/.cache/huggingface/datasets/
```

**Pros:**
- Automatic download and versioning
- Built-in preprocessing tools
- Easy splitting and sampling
- Automatic caching

### Method 2: Manual GitHub Download

**For datasets not on HuggingFace:**

```bash
# WinoBias
git clone https://github.com/uclanlp/winobias.git
cd winobias
python read_data.py

# StereoSet
git clone https://github.com/moinnadeem/stereoset.git
cd stereoset
python stereoset.py --load-data

# TruthfulQA
git clone https://github.com/sylinrl/TruthfulQA.git
cd TruthfulQA
# Data already included

# FEVER
git clone https://github.com/fever/fever-code.git
cd fever-code
bash download_data.sh
```

### Method 3: Direct Download Links

```bash
# Create data directory
mkdir -p ./datasets
cd datasets

# BOLD Dataset
wget https://huggingface.co/datasets/facebook/bold/raw/main/bold/data.tar.gz
tar -xzf data.tar.gz

# TruthfulQA (from Hugging Face)
wget https://huggingface.co/datasets/truthful_qa/resolve/main/TruthfulQA.csv

# SQuAD
wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v2.0.json
wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json
```

---

## Practical Example: Loading & Inspecting Data

```python
from datasets import load_dataset
import pandas as pd

# ========== LOAD DATASETS ==========

# 1. BIAS DETECTION
print("Loading BOLD dataset...")
bold = load_dataset('facebook/bold')
print(f"BOLD: {bold['train'].num_rows} samples")
print(f"Features: {bold['train'].features}")
print(f"Sample: {bold['train'][0]}")

# 2. HALLUCINATION DETECTION
print("\nLoading TruthfulQA...")
truthful_qa = load_dataset('truthful_qa', 'generation')
print(f"TruthfulQA: {truthful_qa['validation'].num_rows} samples")
print(f"Sample question: {truthful_qa['validation'][0]['question']}")

print("\nLoading FEVER...")
fever = load_dataset('fever')
print(f"FEVER: {fever['train'].num_rows} samples")
print(f"Sample claim: {fever['train'][0]['claim']}")

print("\nLoading SQuAD...")
squad = load_dataset('squad')
print(f"SQuAD: {squad['train'].num_rows} samples")

# 3. EXPLAINABILITY
print("\nLoading e-SNLI...")
e_snli = load_dataset('e_snli')
print(f"e-SNLI: {e_snli['train'].num_rows} samples")

# ========== BASIC PREPROCESSING ==========

# Convert to pandas for easier manipulation
def to_pandas(dataset):
    return pd.DataFrame(dataset)

bold_df = to_pandas(bold['train'].select(range(1000)))  # Sample 1000
print(f"\nBOLD DataFrame shape: {bold_df.shape}")
print(bold_df.head())

# ========== STATISTICS ==========

print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)

datasets_summary = {
    'BOLD': bold['train'].num_rows,
    'TruthfulQA': truthful_qa['validation'].num_rows,
    'FEVER': fever['train'].num_rows,
    'SQuAD': squad['train'].num_rows,
    'e-SNLI': e_snli['train'].num_rows,
}

for name, count in datasets_summary.items():
    print(f"{name:20} : {count:,} samples")

total = sum(datasets_summary.values())
print(f"\n{'Total':20} : {total:,} samples")
```

---

## Organizing Downloaded Data

```
project_root/
├── data/
│   ├── bias/
│   │   ├── bold/
│   │   │   └── data.json
│   │   ├── stereoset/
│   │   │   └── data.json
│   │   └── winobias/
│   │       └── data.json
│   │
│   ├── hallucination/
│   │   ├── truthful_qa/
│   │   │   └── TruthfulQA.csv
│   │   ├── fever/
│   │   │   └── fever_data.json
│   │   ├── hotpot_qa/
│   │   │   └── hotpot_data.json
│   │   └── squad/
│   │       ├── train-v2.0.json
│   │       └── dev-v2.0.json
│   │
│   └── explainability/
│       ├── e_snli/
│       │   └── snli_1.0.json
│       └── eraser/
│           └── eraser_data.json
│
├── dataset_loader.py
├── main.py
└── requirements.txt
```

---

## Complete Setup Script

Save as `setup_datasets.py`:

```python
#!/usr/bin/env python3
"""
Automated script to download and organize all datasets
"""

import os
from pathlib import Path
from datasets import load_dataset
import json

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
    
    print("✓ Directory structure created")

def download_datasets():
    """Download all datasets"""
    
    print("\n" + "="*60)
    print("DOWNLOADING DATASETS")
    print("="*60)
    
    # Bias datasets
    print("\n1. BIAS DETECTION DATASETS")
    print("-" * 40)
    
    try:
        print("  Downloading BOLD...")
        bold = load_dataset('facebook/bold')
        print("  ✓ BOLD downloaded")
    except Exception as e:
        print(f"  ✗ BOLD failed: {e}")
    
    try:
        print("  Downloading e-SNLI...")
        e_snli = load_dataset('e_snli')
        print("  ✓ e-SNLI downloaded")
    except Exception as e:
        print(f"  ✗ e-SNLI failed: {e}")
    
    # Hallucination datasets
    print("\n2. HALLUCINATION DETECTION DATASETS")
    print("-" * 40)
    
    try:
        print("  Downloading TruthfulQA...")
        truthful_qa = load_dataset('truthful_qa', 'generation')
        print("  ✓ TruthfulQA downloaded")
    except Exception as e:
        print(f"  ✗ TruthfulQA failed: {e}")
    
    try:
        print("  Downloading FEVER...")
        fever = load_dataset('fever')
        print("  ✓ FEVER downloaded")
    except Exception as e:
        print(f"  ✗ FEVER failed: {e}")
    
    try:
        print("  Downloading SQuAD...")
        squad = load_dataset('squad')
        print("  ✓ SQuAD downloaded")
    except Exception as e:
        print(f"  ✗ SQuAD failed: {e}")
    
    try:
        print("  Downloading HotpotQA...")
        hotpot = load_dataset('hotpot_qa', 'distractor')
        print("  ✓ HotpotQA downloaded")
    except Exception as e:
        print(f"  ✗ HotpotQA failed: {e}")
    
    print("\n" + "="*60)
    print("✓ Dataset download complete!")
    print("="*60)
    print("\nNote: Datasets are cached in ~/.cache/huggingface/datasets/")

if __name__ == "__main__":
    print("Setting up Explainable AI System datasets...")
    create_directory_structure()
    download_datasets()
    print("\n✓ Setup complete! You can now use the datasets.")
```

Run it:
```bash
python setup_datasets.py
```

---

## Quick Run Commands

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download and organize datasets
python setup_datasets.py

# Step 3: Load and explore datasets
python dataset_loader.py
```

---

## Dataset Summary Table

| Dataset | Type | Size | Download | License |
|---------|------|------|----------|---------|
| **BOLD** | Bias | 23K | `load_dataset('facebook/bold')` | CC-BY-4.0 |
| **TruthfulQA** | Hallucination | 817 | `load_dataset('truthful_qa', 'generation')` | MIT |
| **FEVER** | Hallucination | 185K | `load_dataset('fever')` | CC-BY-4.0 |
| **SQuAD** | Hallucination | 100K | `load_dataset('squad')` | CC-BY-SA-4.0 |
| **HotpotQA** | Hallucination | 113K | `load_dataset('hotpot_qa', 'distractor')` | CC-BY-4.0 |
| **e-SNLI** | Explainability | 570K | `load_dataset('e_snli')` | CC-BY-4.0 |
| **StereoSet** | Bias | 17K | GitHub or HuggingFace | CC-BY-4.0 |
| **WinoBias** | Bias | 3.2K | GitHub | MIT |

---

## Troubleshooting

### Issue: Connection timeout when downloading

**Solution:**
```python
# Set timeout and retry
import huggingface_hub
huggingface_hub.constants.INFERENCE_API_TIMEOUT = 300

# Or download manually from GitHub
git clone <repository_url>
```

### Issue: Disk space insufficient

**Solution:**
```python
# Load only specific splits
dataset = load_dataset('dataset_name', split='train[:10%]')  # 10% sample
```

### Issue: Slow downloads

**Solution:**
```bash
# Use parallel downloading
pip install aria2
# Configure HuggingFace Hub for parallel downloads
```

### Issue: Dataset not found on HuggingFace

**Solution:**
- Check the official dataset repository on GitHub
- Look for alternative dataset mirrors
- Consider creating a custom dataset

---

## Storage Space Estimates

- **BOLD**: ~500 MB
- **TruthfulQA**: ~5 MB
- **FEVER**: ~3 GB
- **SQuAD**: ~500 MB
- **HotpotQA**: ~2 GB
- **e-SNLI**: ~8 GB
- **All Other Datasets**: ~2 GB

**Total: ~16 GB** (approximately)

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Download datasets using the provided script
3. ✅ Use `dataset_loader.py` for loading and preprocessing
4. ✅ Explore dataset structure and statistics
5. → Begin model training and evaluation
