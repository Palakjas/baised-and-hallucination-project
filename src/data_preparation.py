"""
Data Preparation Module
Handles loading, cleaning, and preprocessing of datasets
for the XAI Bias & Hallucination detection pipeline.
"""

import os
import json
import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Prepares and preprocesses data for the XAI pipeline.
    Handles text cleaning, splitting, and formatting for
    bias detection, hallucination detection, and explainability tasks.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize data preparation.
        
        Args:
            data_dir: Root directory for data storage.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataPreparation initialized with data_dir: {self.data_dir}")

    # ==================== TEXT CLEANING ====================

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text.
            
        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,;:!?\'\"()\-]', '', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text.
            
        Returns:
            List of sentence strings.
        """
        abbreviations = ["Mr.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "e.g.", "i.e.", "etc."]
        temp = text
        for abbr in abbreviations:
            temp = temp.replace(abbr, abbr.replace(".", "<DOT>"))

        sentences = re.split(r'(?<=[.!?])\s+', temp)
        return [s.replace("<DOT>", ".").strip() for s in sentences if len(s.strip()) > 3]

    # ==================== DATASET FORMATTING ====================

    def format_for_bias_detection(self, texts: List[str], labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Format data for bias detection tasks.
        
        Args:
            texts: List of text samples.
            labels: Optional list of bias labels.
            
        Returns:
            Formatted DataFrame.
        """
        data = {
            "text": [self.clean_text(t) for t in texts],
            "text_length": [len(t.split()) for t in texts],
        }

        if labels:
            data["label"] = labels

        df = pd.DataFrame(data)
        df = df[df["text"].str.len() > 0]  # Remove empty texts
        logger.info(f"Formatted {len(df)} samples for bias detection.")
        return df

    def format_for_hallucination_detection(
        self,
        claims: List[str],
        contexts: List[str],
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Format data for hallucination/fact verification tasks.
        
        Args:
            claims: List of claims to verify.
            contexts: List of reference contexts.
            labels: Optional list of labels (SUPPORTS, REFUTES, NOT_ENOUGH_INFO).
            
        Returns:
            Formatted DataFrame.
        """
        data = {
            "claim": [self.clean_text(c) for c in claims],
            "context": [self.clean_text(ctx) for ctx in contexts],
        }

        if labels:
            data["label"] = labels

        df = pd.DataFrame(data)
        df = df[(df["claim"].str.len() > 0) & (df["context"].str.len() > 0)]
        logger.info(f"Formatted {len(df)} samples for hallucination detection.")
        return df

    def format_for_explainability(
        self,
        premises: List[str],
        hypotheses: List[str],
        labels: List[int],
        explanations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Format data for explainability tasks (e.g., e-SNLI style).
        
        Args:
            premises: List of premise sentences.
            hypotheses: List of hypothesis sentences.
            labels: List of labels (0=entailment, 1=neutral, 2=contradiction).
            explanations: Optional list of human explanations.
            
        Returns:
            Formatted DataFrame.
        """
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        data = {
            "premise": [self.clean_text(p) for p in premises],
            "hypothesis": [self.clean_text(h) for h in hypotheses],
            "label": [label_map.get(l, str(l)) for l in labels],
        }

        if explanations:
            data["explanation"] = [self.clean_text(e) for e in explanations]

        df = pd.DataFrame(data)
        logger.info(f"Formatted {len(df)} samples for explainability.")
        return df

    # ==================== TRAIN/VAL/TEST SPLITTING ====================

    def create_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a DataFrame into train/validation/test sets.
        
        Args:
            df: Input DataFrame.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = df.iloc[indices[:train_end]].reset_index(drop=True)
        val_df = df.iloc[indices[train_end:val_end]].reset_index(drop=True)
        test_df = df.iloc[indices[val_end:]].reset_index(drop=True)

        logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    # ==================== SAVE / LOAD ====================

    def save_dataset(self, df: pd.DataFrame, name: str, fmt: str = "json") -> str:
        """
        Save a DataFrame to disk.
        
        Args:
            df: DataFrame to save.
            name: File name (without extension).
            fmt: Format - 'json' or 'csv'.
            
        Returns:
            Path to saved file.
        """
        path = self.data_dir / f"{name}.{fmt}"
        if fmt == "json":
            df.to_json(path, orient="records", indent=2)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info(f"Saved dataset to {path}")
        return str(path)

    def load_dataset(self, name: str, fmt: str = "json") -> pd.DataFrame:
        """
        Load a dataset from disk.
        
        Args:
            name: File name (without extension).
            fmt: Format - 'json' or 'csv'.
            
        Returns:
            Loaded DataFrame.
        """
        path = self.data_dir / f"{name}.{fmt}"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if fmt == "json":
            df = pd.read_json(path)
        elif fmt == "csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info(f"Loaded dataset from {path} ({len(df)} rows)")
        return df

    # ==================== STATISTICS ====================

    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get basic statistics about a dataset.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Dict with statistics.
        """
        stats = {
            "num_samples": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
        }

        # Text column stats
        for col in df.columns:
            if df[col].dtype == "object":
                lengths = df[col].astype(str).str.split().str.len()
                stats[f"{col}_avg_words"] = round(float(lengths.mean()), 1)
                stats[f"{col}_max_words"] = int(lengths.max())

        return stats
