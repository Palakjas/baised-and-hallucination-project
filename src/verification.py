"""
Fact Verification Module
Verifies extracted atomic facts against a trusted knowledge base
using semantic similarity (sentence-transformers).
"""

import logging
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class FactVerifier:
    """
    Verifies facts against a knowledge base using semantic similarity.
    
    Uses sentence-transformers to encode facts and knowledge base sentences,
    then computes cosine similarity to determine support.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.5):
        """
        Initialize the verifier.
        
        Args:
            model_name: Sentence-transformer model for embeddings.
            threshold: Minimum cosine similarity to consider a fact "supported".
        """
        self.model_name = model_name
        self.threshold = threshold

        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"FactVerifier initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _split_context(self, context: str) -> List[str]:
        """Split context into individual sentences for comparison."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', context.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def verify_fact(self, fact: str, context: str) -> Dict[str, Any]:
        """
        Verify a single fact against a context/knowledge base.
        
        Args:
            fact: The atomic fact to verify.
            context: The trusted knowledge base text.
            
        Returns:
            Dict with verification results:
                - is_supported (bool)
                - confidence (float)
                - evidence (str): best matching sentence
                - similarity_score (float)
        """
        context_sentences = self._split_context(context)

        if not context_sentences:
            return {
                "is_supported": False,
                "confidence": 0.0,
                "evidence": "No context provided.",
                "similarity_score": 0.0,
            }

        # Encode fact and context sentences
        fact_embedding = self.model.encode(fact, convert_to_tensor=True)
        context_embeddings = self.model.encode(context_sentences, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.cos_sim(fact_embedding, context_embeddings)[0]
        best_idx = int(np.argmax(similarities.cpu().numpy()))
        best_score = float(similarities[best_idx].cpu().numpy())

        is_supported = best_score >= self.threshold

        return {
            "is_supported": is_supported,
            "confidence": best_score,
            "evidence": context_sentences[best_idx],
            "similarity_score": best_score,
        }

    def verify_response(self, facts: List[str], context: str) -> Dict[str, Any]:
        """
        Verify a list of facts against a knowledge base context.
        This is the main method called by app.py.
        
        Args:
            facts: List of atomic fact strings.
            context: Trusted knowledge base text.
            
        Returns:
            Dict with structure:
                {
                    "facts": [
                        {
                            "fact": str,
                            "verification": {
                                "is_supported": bool,
                                "confidence": float,
                                "evidence": str,
                            }
                        },
                        ...
                    ],
                    "summary": {
                        "total_facts": int,
                        "supported": int,
                        "not_supported": int,
                        "support_rate": float,
                    }
                }
        """
        results = []
        supported_count = 0

        for fact in facts:
            verification = self.verify_fact(fact, context)
            results.append({
                "fact": fact,
                "verification": verification,
            })
            if verification["is_supported"]:
                supported_count += 1

        total = len(facts)
        summary = {
            "total_facts": total,
            "supported": supported_count,
            "not_supported": total - supported_count,
            "support_rate": supported_count / total if total > 0 else 0.0,
        }

        logger.info(
            f"Verification complete: {supported_count}/{total} facts supported "
            f"({summary['support_rate']:.1%})"
        )

        return {"facts": results, "summary": summary}

    def batch_verify(self, fact_sets: List[Dict], context: str) -> List[Dict]:
        """
        Verify multiple sets of facts (batch mode).
        
        Args:
            fact_sets: List of dicts with 'facts' key containing list of fact strings.
            context: Trusted knowledge base text.
            
        Returns:
            List of verification results for each fact set.
        """
        return [
            self.verify_response(fs["facts"], context)
            for fs in fact_sets
        ]
