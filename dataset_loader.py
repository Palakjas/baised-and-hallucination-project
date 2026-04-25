"""
Dataset Loader for Explainable AI System: Detecting Bias & Hallucination in LLMs
This script helps download, load, and preprocess all necessary datasets
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BiasHallucinationDatasetLoader:
    """
    Unified loader for bias detection, hallucination detection, and explainability datasets
    """
    
    def __init__(self, cache_dir='./data'):
        """
        Initialize the dataset loader
        
        Args:
            cache_dir (str): Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        
    def load_bias_datasets(self):
        """Load all bias detection datasets"""
        logger.info("Loading bias detection datasets...")
        
        bias_datasets = {
            'bold': self._load_bold,
            'stereoset': self._load_stereoset,
            'winobias': self._load_winobias,
            'bias_in_bios': self._load_bias_in_bios,
        }
        
        for name, loader_func in bias_datasets.items():
            try:
                logger.info(f"  Loading {name}...")
                self.datasets[name] = loader_func()
                logger.info(f"  ✓ {name} loaded successfully")
            except Exception as e:
                logger.error(f"  ✗ Error loading {name}: {e}")
        
        return self.datasets
    
    def load_hallucination_datasets(self):
        """Load all hallucination detection datasets"""
        logger.info("Loading hallucination detection datasets...")
        
        halluc_datasets = {
            'truthful_qa': self._load_truthful_qa,
            'fever': self._load_fever,
            'hotpot_qa': self._load_hotpot_qa,
            'squad': self._load_squad,
        }
        
        for name, loader_func in halluc_datasets.items():
            try:
                logger.info(f"  Loading {name}...")
                self.datasets[name] = loader_func()
                logger.info(f"  ✓ {name} loaded successfully")
            except Exception as e:
                logger.error(f"  ✗ Error loading {name}: {e}")
        
        return self.datasets
    
    def load_explainability_datasets(self):
        """Load all explainability datasets"""
        logger.info("Loading explainability datasets...")
        
        explain_datasets = {
            'e_snli': self._load_e_snli,
            'eraser': self._load_eraser,
        }
        
        for name, loader_func in explain_datasets.items():
            try:
                logger.info(f"  Loading {name}...")
                self.datasets[name] = loader_func()
                logger.info(f"  ✓ {name} loaded successfully")
            except Exception as e:
                logger.error(f"  ✗ Error loading {name}: {e}")
        
        return self.datasets
    
    # ==================== BIAS DATASETS ====================
    
    def _load_bold(self):
        """Load BOLD dataset (Bias in Open-ended Language Generation)"""
        dataset = load_dataset('facebook/bold', cache_dir=str(self.cache_dir))
        return dataset
    
    def _load_stereoset(self):
        """Load StereoSet dataset"""
        try:
            dataset = load_dataset('stereoset', 'intrasentence', cache_dir=str(self.cache_dir))
            return dataset
        except:
            logger.warning("StereoSet not available via HuggingFace, providing alternative...")
            return None
    
    def _load_winobias(self):
        """Load WinoBias dataset"""
        try:
            dataset = load_dataset('wino_bias', cache_dir=str(self.cache_dir))
            return dataset
        except:
            logger.warning("WinoBias might need manual download from GitHub")
            return None
    
    def _load_bias_in_bios(self):
        """Load Bias in Bios dataset"""
        try:
            dataset = load_dataset('bias_in_bios', cache_dir=str(self.cache_dir))
            return dataset
        except:
            logger.warning("Bias in Bios might need manual download from GitHub")
            return None
    
    # ==================== HALLUCINATION DATASETS ====================
    
    def _load_truthful_qa(self):
        """Load TruthfulQA dataset"""
        dataset = load_dataset('truthful_qa', 'generation', cache_dir=str(self.cache_dir))
        return dataset
    
    def _load_fever(self):
        """Load FEVER dataset (Fact Extraction and VERification)"""
        dataset = load_dataset('fever', 'wikipedia_pages', cache_dir=str(self.cache_dir))
        return dataset
    
    def _load_hotpot_qa(self):
        """Load HotpotQA dataset"""
        dataset = load_dataset('hotpot_qa', 'distractor', cache_dir=str(self.cache_dir))
        return dataset
    
    def _load_squad(self):
        """Load SQuAD dataset"""
        dataset = load_dataset('squad', cache_dir=str(self.cache_dir))
        return dataset
    
    # ==================== EXPLAINABILITY DATASETS ====================
    
    def _load_e_snli(self):
        """Load e-SNLI dataset (NLI with explanations)"""
        dataset = load_dataset('e_snli', cache_dir=str(self.cache_dir))
        return dataset
    
    def _load_eraser(self):
        """Load ERASER benchmark"""
        try:
            dataset = load_dataset('eraser', cache_dir=str(self.cache_dir))
            return dataset
        except:
            logger.warning("ERASER might need special handling")
            return None
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def explore_dataset(self, dataset_name):
        """
        Explore structure and content of a loaded dataset
        
        Args:
            dataset_name (str): Name of the dataset to explore
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not loaded")
            return
        
        dataset = self.datasets[dataset_name]
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        if isinstance(dataset, DatasetDict):
            print(f"Splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                print(f"\n  {split_name}:")
                print(f"    Size: {len(split_data)}")
                print(f"    Features: {split_data.features.keys()}")
                print(f"    Sample:")
                print(f"      {split_data[0]}")
        else:
            print(f"Size: {len(dataset)}")
            print(f"Features: {dataset.features.keys()}")
            print(f"Sample: {dataset[0]}")
    
    def create_combined_dataset(self, output_path='./combined_dataset.json'):
        """
        Create a combined dataset with all loaded datasets
        
        Args:
            output_path (str): Path to save combined dataset
        """
        logger.info(f"Creating combined dataset at {output_path}...")
        
        combined = {
            'bias_samples': [],
            'hallucination_samples': [],
            'explainability_samples': [],
            'metadata': {
                'total_bias_samples': 0,
                'total_hallucination_samples': 0,
                'total_explainability_samples': 0,
            }
        }
        
        # Process bias datasets
        bias_keys = ['bold', 'stereoset', 'winobias', 'bias_in_bios']
        for key in bias_keys:
            if key in self.datasets and self.datasets[key]:
                try:
                    samples = self._extract_bias_samples(key, self.datasets[key])
                    combined['bias_samples'].extend(samples)
                    combined['metadata']['total_bias_samples'] += len(samples)
                except Exception as e:
                    logger.warning(f"Could not extract samples from {key}: {e}")
        
        # Process hallucination datasets
        halluc_keys = ['truthful_qa', 'fever', 'hotpot_qa', 'squad']
        for key in halluc_keys:
            if key in self.datasets and self.datasets[key]:
                try:
                    samples = self._extract_hallucination_samples(key, self.datasets[key])
                    combined['hallucination_samples'].extend(samples)
                    combined['metadata']['total_hallucination_samples'] += len(samples)
                except Exception as e:
                    logger.warning(f"Could not extract samples from {key}: {e}")
        
        # Process explainability datasets
        explain_keys = ['e_snli', 'eraser']
        for key in explain_keys:
            if key in self.datasets and self.datasets[key]:
                try:
                    samples = self._extract_explainability_samples(key, self.datasets[key])
                    combined['explainability_samples'].extend(samples)
                    combined['metadata']['total_explainability_samples'] += len(samples)
                except Exception as e:
                    logger.warning(f"Could not extract samples from {key}: {e}")
        
        # Save combined dataset
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2)
        
        logger.info(f"Combined dataset saved to {output_path}")
        logger.info(f"  - Bias samples: {combined['metadata']['total_bias_samples']}")
        logger.info(f"  - Hallucination samples: {combined['metadata']['total_hallucination_samples']}")
        logger.info(f"  - Explainability samples: {combined['metadata']['total_explainability_samples']}")
        
        return combined
    
    def _extract_bias_samples(self, dataset_name, dataset, limit=1000):
        """Extract relevant samples from bias datasets"""
        samples = []
        
        if dataset_name == 'bold':
            if isinstance(dataset, DatasetDict):
                data = dataset['train']
            else:
                data = dataset
            
            for i, sample in enumerate(data):
                if i >= limit:
                    break
                samples.append({
                    'source': 'bold',
                    'text': sample.get('text', ''),
                    'category': sample.get('category', ''),
                    'target': sample.get('target', ''),
                })
        
        elif dataset_name == 'stereoset':
            if isinstance(dataset, DatasetDict):
                data = dataset['intrasentence']
            else:
                data = dataset
            
            for i, sample in enumerate(data):
                if i >= limit:
                    break
                samples.append({
                    'source': 'stereoset',
                    'context': sample.get('context', ''),
                    'target': sample.get('target', ''),
                    'stereotype': sample.get('stereotype', ''),
                })
        
        return samples
    
    def _extract_hallucination_samples(self, dataset_name, dataset, limit=1000):
        """Extract relevant samples from hallucination datasets"""
        samples = []
        
        if dataset_name == 'truthful_qa':
            if isinstance(dataset, DatasetDict):
                data = dataset['validation']
            else:
                data = dataset
            
            for i, sample in enumerate(data):
                if i >= limit:
                    break
                samples.append({
                    'source': 'truthful_qa',
                    'question': sample.get('question', ''),
                    'answers': sample.get('correct_answers', []),
                    'incorrect_answers': sample.get('incorrect_answers', []),
                })
        
        elif dataset_name == 'fever':
            if isinstance(dataset, DatasetDict):
                data = dataset.get('train', dataset.get('validation', dataset))
            else:
                data = dataset
            
            for i, sample in enumerate(data):
                if i >= limit:
                    break
                samples.append({
                    'source': 'fever',
                    'claim': sample.get('claim', ''),
                    'label': sample.get('label', ''),
                    'evidence': sample.get('evidence', ''),
                })
        
        return samples
    
    def _extract_explainability_samples(self, dataset_name, dataset, limit=500):
        """Extract relevant samples from explainability datasets"""
        samples = []
        
        if dataset_name == 'e_snli':
            if isinstance(dataset, DatasetDict):
                data = dataset['train']
            else:
                data = dataset
            
            for i, sample in enumerate(data):
                if i >= limit:
                    break
                samples.append({
                    'source': 'e_snli',
                    'premise': sample.get('premise', ''),
                    'hypothesis': sample.get('hypothesis', ''),
                    'label': sample.get('label', ''),
                    'explanation': sample.get('explanation_1', ''),
                })
        
        return samples
    
    def get_statistics(self):
        """Get statistics about loaded datasets"""
        stats = {}
        
        for name, dataset in self.datasets.items():
            if dataset is None:
                stats[name] = "Not loaded"
                continue
            
            if isinstance(dataset, DatasetDict):
                stats[name] = {
                    'splits': list(dataset.keys()),
                    'total_samples': sum(len(split) for split in dataset.values()),
                    'features': list(dataset['train'].features.keys()) if 'train' in dataset else [],
                }
            else:
                stats[name] = {
                    'total_samples': len(dataset),
                    'features': list(dataset.features.keys()),
                }
        
        return stats
    
    def save_statistics(self, output_path='./dataset_statistics.json'):
        """Save dataset statistics to file"""
        stats = self.get_statistics()
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {output_path}")


# ==================== EXAMPLE USAGE ====================

def main():
    """Main execution function"""
    
    # Initialize loader
    loader = BiasHallucinationDatasetLoader(cache_dir='./nlp_data')
    
    # Load all datasets
    logger.info("Starting dataset loading process...")
    
    # Load bias datasets
    logger.info("\n" + "="*60)
    loader.load_bias_datasets()
    
    # Load hallucination datasets
    logger.info("\n" + "="*60)
    loader.load_hallucination_datasets()
    
    # Load explainability datasets
    logger.info("\n" + "="*60)
    loader.load_explainability_datasets()
    
    # Explore datasets
    logger.info("\n" + "="*60)
    logger.info("Exploring loaded datasets...\n")
    for dataset_name in loader.datasets.keys():
        if loader.datasets[dataset_name]:
            loader.explore_dataset(dataset_name)
    
    # Get statistics
    logger.info("\n" + "="*60)
    logger.info("Dataset Statistics:")
    stats = loader.get_statistics()
    for name, stat in stats.items():
        print(f"\n{name}:")
        if isinstance(stat, dict):
            for key, value in stat.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {stat}")
    
    # Save statistics
    loader.save_statistics('./dataset_statistics.json')
    
    # Create combined dataset
    logger.info("\n" + "="*60)
    combined = loader.create_combined_dataset('./combined_dataset.json')
    
    logger.info("\n" + "="*60)
    logger.info("✓ Dataset loading complete!")


if __name__ == "__main__":
    main()
