"""
Example Implementation: Bias and Hallucination Detection Modules
Demonstrates how to use the datasets for model development
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Tuple
import json

# ==================== BIAS DETECTION MODULE ====================

class BiasDetectionModule:
    """Detect and analyze biases in LLM outputs"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize bias detection module
        
        Args:
            model_name: Pretrained model from HuggingFace
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load BOLD dataset for bias evaluation
        self.bold_dataset = load_dataset('facebook/bold')['train']
        
    def detect_gender_bias(self, text: str) -> Dict[str, float]:
        """
        Detect gender bias in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with gender bias scores
        """
        # Define gender-associated terms
        gender_terms = {
            'masculine': ['he', 'him', 'his', 'man', 'boy', 'male', 'king', 'prince'],
            'feminine': ['she', 'her', 'hers', 'woman', 'girl', 'female', 'queen', 'princess'],
            'neutral': ['they', 'them', 'theirs', 'person', 'human', 'one', 'individual']
        }
        
        text_lower = text.lower()
        
        # Count gender-associated terms
        scores = {}
        for gender, terms in gender_terms.items():
            count = sum(text_lower.count(term) for term in terms)
            scores[gender] = count
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            normalized_scores = {k: v/total for k, v in scores.items()}
        else:
            normalized_scores = {k: 0 for k in scores.keys()}
        
        return normalized_scores
    
    def detect_occupational_bias(self, occupation: str, adjectives: List[str]) -> Dict[str, float]:
        """
        Detect bias in occupational descriptions
        
        Args:
            occupation: Job title or occupation
            adjectives: List of adjectives used to describe the occupation
            
        Returns:
            Bias score dictionary
        """
        # Pre-defined stereotypical adjectives by occupation
        stereotypes = {
            'nurse': ['caring', 'compassionate', 'nurturing', 'helpful'],
            'engineer': ['smart', 'technical', 'analytical', 'logical'],
            'secretary': ['organized', 'efficient', 'detail-oriented'],
            'ceo': ['ambitious', 'decisive', 'powerful', 'confident'],
        }
        
        bias_score = 0
        if occupation.lower() in stereotypes:
            stereotype_adjectives = stereotypes[occupation.lower()]
            matching = sum(1 for adj in adjectives if adj.lower() in stereotype_adjectives)
            bias_score = matching / len(adjectives) if adjectives else 0
        
        return {
            'occupation': occupation,
            'stereotype_match_ratio': bias_score,
            'is_biased': bias_score > 0.5
        }
    
    def evaluate_stereotypes(self, text: str) -> Dict[str, any]:
        """
        Evaluate stereotypical content using BOLD dataset
        
        Args:
            text: Input text to evaluate
            
        Returns:
            Detailed bias analysis
        """
        # Extract gender, occupational, and demographic biases
        analysis = {
            'gender_bias': self.detect_gender_bias(text),
            'text_length': len(text.split()),
            'has_stereotypes': self._check_stereotypes(text),
        }
        
        return analysis
    
    def _check_stereotypes(self, text: str) -> bool:
        """Check if text contains known stereotypes"""
        stereotypes_phrases = [
            'women are emotional',
            'men are strong',
            'asians are good at math',
            'women like cooking',
            'men are breadwinners',
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in stereotypes_phrases)
    
    def generate_bias_report(self, texts: List[str]) -> pd.DataFrame:
        """
        Generate comprehensive bias report for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with bias analysis
        """
        results = []
        
        for i, text in enumerate(texts):
            bias_analysis = self.evaluate_stereotypes(text)
            
            results.append({
                'text_id': i,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'gender_bias_masculine': bias_analysis['gender_bias'].get('masculine', 0),
                'gender_bias_feminine': bias_analysis['gender_bias'].get('feminine', 0),
                'gender_bias_neutral': bias_analysis['gender_bias'].get('neutral', 0),
                'has_stereotypes': bias_analysis['has_stereotypes'],
            })
        
        return pd.DataFrame(results)


# ==================== HALLUCINATION DETECTION MODULE ====================

class HallucinationDetectionModule:
    """Detect hallucinations in LLM outputs"""
    
    def __init__(self):
        """Initialize hallucination detection module"""
        # Load hallucination datasets
        self.truthful_qa = load_dataset('truthful_qa', 'generation')['validation']
        self.fever_data = load_dataset('fever')['train']
        
        # Initialize fact verification pipeline
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    
    def check_factual_consistency(self, text: str, facts: List[str]) -> Dict[str, float]:
        """
        Check if generated text is consistent with known facts
        
        Args:
            text: Generated text to verify
            facts: List of ground truth facts
            
        Returns:
            Consistency scores
        """
        results = {}
        
        for fact in facts:
            # Use zero-shot classification to check consistency
            result = self.zero_shot_classifier(
                text,
                [f"This statement is consistent with: {fact}",
                 f"This statement contradicts: {fact}"]
            )
            
            # Extract score
            consistency_score = result['scores'][0]
            results[fact] = consistency_score
        
        return results
    
    def detect_contradiction(self, statement1: str, statement2: str) -> float:
        """
        Detect if two statements contradict each other
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            Contradiction score (0-1)
        """
        result = self.zero_shot_classifier(
            statement1,
            [f"This contradicts: {statement2}",
             f"This is consistent with: {statement2}"]
        )
        
        # Higher score = more contradiction
        contradiction_score = result['scores'][0]
        return contradiction_score
    
    def check_against_context(self, generated_text: str, context: str) -> Dict[str, any]:
        """
        Check if generated text is grounded in provided context
        
        Args:
            generated_text: Text generated by model
            context: Source context/passage
            
        Returns:
            Grounding analysis
        """
        # Check if key entities from generated text appear in context
        from collections import Counter
        
        generated_words = set(generated_text.lower().split())
        context_words = set(context.lower().split())
        
        overlap = generated_words.intersection(context_words)
        grounding_score = len(overlap) / len(generated_words) if generated_words else 0
        
        return {
            'grounding_score': grounding_score,
            'is_grounded': grounding_score > 0.5,
            'overlapping_entities': list(overlap)[:10],  # Top 10
        }
    
    def evaluate_answer_quality(self, question: str, answer: str, 
                               reference_answers: List[str]) -> Dict[str, any]:
        """
        Evaluate if answer matches reference answers
        
        Args:
            question: Input question
            answer: Generated answer
            reference_answers: Ground truth answers
            
        Returns:
            Quality evaluation
        """
        # Simple word overlap metric
        answer_words = set(answer.lower().split())
        
        max_overlap = 0
        best_match = None
        
        for ref_answer in reference_answers:
            ref_words = set(ref_answer.lower().split())
            overlap = len(answer_words.intersection(ref_words)) / max(len(answer_words), 1)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = ref_answer
        
        return {
            'answer_match_score': max_overlap,
            'is_hallucinated': max_overlap < 0.3,
            'best_matching_reference': best_match,
            'confidence': max_overlap,
        }
    
    def generate_hallucination_report(self, qa_pairs: List[Dict]) -> pd.DataFrame:
        """
        Generate hallucination detection report
        
        Args:
            qa_pairs: List of dicts with 'question', 'answer', 'reference_answers'
            
        Returns:
            DataFrame with hallucination analysis
        """
        results = []
        
        for i, pair in enumerate(qa_pairs):
            quality = self.evaluate_answer_quality(
                pair['question'],
                pair['answer'],
                pair.get('reference_answers', [])
            )
            
            results.append({
                'qa_id': i,
                'question': pair['question'][:100],
                'answer': pair['answer'][:100],
                'match_score': quality['answer_match_score'],
                'is_hallucinated': quality['is_hallucinated'],
                'confidence': quality['confidence'],
            })
        
        return pd.DataFrame(results)


# ==================== EXPLAINABILITY MODULE ====================

class ExplainabilityModule:
    """Generate explanations for model decisions"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize explainability module"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Load e-SNLI for explanation training
        self.e_snli = load_dataset('e_snli')['train']
    
    def extract_attention_weights(self, text: str) -> np.ndarray:
        """
        Extract attention weights from model
        
        Args:
            text: Input text
            
        Returns:
            Attention weight matrix
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        
        # Average attention across heads and layers
        avg_attention = torch.mean(attentions[0], dim=1).cpu().numpy()
        return avg_attention
    
    def identify_important_tokens(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Identify most important tokens for model decision
        
        Args:
            text: Input text
            top_k: Number of top tokens to return
            
        Returns:
            List of (token, importance_score) tuples
        """
        attention_weights = self.extract_attention_weights(text)
        tokens = self.tokenizer.tokenize(text)
        
        # Use average attention weight as importance
        importance_scores = attention_weights[0][:len(tokens)]
        
        # Get top-k tokens
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        important_tokens = [(tokens[i], importance_scores[i]) for i in top_indices]
        
        return important_tokens
    
    def generate_text_explanation(self, text: str, label: int) -> str:
        """
        Generate human-readable explanation
        
        Args:
            text: Input text
            label: Predicted label
            
        Returns:
            Natural language explanation
        """
        important_tokens = self.identify_important_tokens(text, top_k=3)
        
        token_names = [token for token, score in important_tokens]
        explanation = f"The model predicted label {label} based on key terms: {', '.join(token_names)}"
        
        return explanation
    
    def generate_explanation_report(self, texts: List[str], labels: List[int]) -> pd.DataFrame:
        """
        Generate explanation report for multiple predictions
        
        Args:
            texts: List of input texts
            labels: List of predicted labels
            
        Returns:
            DataFrame with explanations
        """
        results = []
        
        for text, label in zip(texts, labels):
            important_tokens = self.identify_important_tokens(text)
            explanation = self.generate_text_explanation(text, label)
            
            results.append({
                'text': text[:100],
                'predicted_label': label,
                'explanation': explanation,
                'important_tokens': ', '.join([t[0] for t in important_tokens[:3]]),
            })
        
        return pd.DataFrame(results)


# ==================== INTEGRATED SYSTEM ====================

class ExplainableAISystem:
    """Integrated system for bias, hallucination, and explainability detection"""
    
    def __init__(self):
        """Initialize all detection modules"""
        self.bias_detector = BiasDetectionModule()
        self.hallucination_detector = HallucinationDetectionModule()
        self.explainability_module = ExplainabilityModule()
    
    def analyze_model_output(self, text: str, facts: List[str] = None) -> Dict:
        """
        Comprehensive analysis of model output
        
        Args:
            text: Model-generated text
            facts: Reference facts for verification
            
        Returns:
            Complete analysis report
        """
        report = {
            'bias_analysis': self.bias_detector.evaluate_stereotypes(text),
            'hallucination_check': self.hallucination_detector.check_against_context(
                text, 
                facts[0] if facts else ""
            ) if facts else None,
            'key_terms': self.explainability_module.identify_important_tokens(text, top_k=5),
        }
        
        return report
    
    def generate_full_report(self, model_outputs: List[str]) -> Dict:
        """
        Generate comprehensive report for multiple outputs
        
        Args:
            model_outputs: List of model-generated texts
            
        Returns:
            Comprehensive system report
        """
        bias_report = self.bias_detector.generate_bias_report(model_outputs)
        
        qa_pairs = [{'question': '', 'answer': output, 'reference_answers': []} 
                   for output in model_outputs]
        halluc_report = self.hallucination_detector.generate_hallucination_report(qa_pairs)
        
        return {
            'bias_analysis': bias_report.to_dict(),
            'hallucination_analysis': halluc_report.to_dict(),
            'summary': {
                'total_outputs': len(model_outputs),
                'avg_bias_score': bias_report['gender_bias_masculine'].mean(),
                'hallucination_rate': halluc_report['is_hallucinated'].sum() / len(halluc_report),
            }
        }


# ==================== EXAMPLE USAGE ====================

def main():
    """Example usage of the system"""
    
    print("Initializing Explainable AI System...")
    system = ExplainableAISystem()
    
    # Example texts for analysis
    example_texts = [
        "The nurse was very caring and compassionate in her work.",
        "The CEO was ambitious and decisive in his leadership decisions.",
        "Paris is the capital of France and is known for its Eiffel Tower.",
    ]
    
    print("\n" + "="*60)
    print("BIAS DETECTION")
    print("="*60)
    
    for i, text in enumerate(example_texts):
        bias_analysis = system.bias_detector.evaluate_stereotypes(text)
        print(f"\nText {i+1}: {text[:50]}...")
        print(f"Gender bias scores: {bias_analysis['gender_bias']}")
        print(f"Contains stereotypes: {bias_analysis['has_stereotypes']}")
    
    print("\n" + "="*60)
    print("HALLUCINATION DETECTION")
    print("="*60)
    
    qa_example = {
        'question': 'What is the capital of France?',
        'answer': 'Paris is the capital of France.',
        'reference_answers': ['Paris', 'The capital of France is Paris']
    }
    
    quality = system.hallucination_detector.evaluate_answer_quality(
        qa_example['question'],
        qa_example['answer'],
        qa_example['reference_answers']
    )
    
    print(f"\nQuestion: {qa_example['question']}")
    print(f"Answer: {qa_example['answer']}")
    print(f"Match Score: {quality['answer_match_score']:.2f}")
    print(f"Is Hallucinated: {quality['is_hallucinated']}")
    
    print("\n" + "="*60)
    print("EXPLAINABILITY")
    print("="*60)
    
    for i, text in enumerate(example_texts[:1]):
        important_tokens = system.explainability_module.identify_important_tokens(text)
        print(f"\nImportant tokens in text {i+1}:")
        for token, score in important_tokens:
            print(f"  {token}: {score:.4f}")
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
