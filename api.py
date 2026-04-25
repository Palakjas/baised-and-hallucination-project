"""
REST API for the XAI Bias & Hallucination Detection System
Provides endpoints for fact extraction, verification, and bias analysis.
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from src.fact_extraction import AtomicFactExtractor
from src.verification import FactVerifier

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Initialize Models ---
logger.info("Loading models...")
verifier = FactVerifier(model_name='all-MiniLM-L6-v2')
extractor = AtomicFactExtractor()
logger.info("Models loaded successfully.")


# ==================== ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "XAI API is running."})


@app.route('/api/extract-facts', methods=['POST'])
def extract_facts():
    """
    Extract atomic facts from text.
    
    Request body:
        {"text": "Your LLM output text here"}
    
    Returns:
        {"facts": ["fact1", "fact2", ...], "count": int}
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body."}), 400

    text = data['text']
    facts = extractor.extract(text)

    return jsonify({
        "facts": facts,
        "count": len(facts),
    })


@app.route('/api/verify', methods=['POST'])
def verify_facts():
    """
    Verify facts against a knowledge base.
    
    Request body:
        {
            "facts": ["fact1", "fact2"],
            "context": "Trusted knowledge base text"
        }
    
    Returns:
        Verification results with similarity scores.
    """
    data = request.get_json()
    if not data or 'facts' not in data or 'context' not in data:
        return jsonify({"error": "Missing 'facts' or 'context' in request body."}), 400

    facts = data['facts']
    context = data['context']

    results = verifier.verify_response(facts, context)
    return jsonify(results)


@app.route('/api/audit', methods=['POST'])
def full_audit():
    """
    Full audit pipeline: extract facts + verify against context.
    
    Request body:
        {
            "text": "LLM generated text",
            "context": "Trusted knowledge base text"
        }
    
    Returns:
        Complete audit results with extraction and verification.
    """
    data = request.get_json()
    if not data or 'text' not in data or 'context' not in data:
        return jsonify({"error": "Missing 'text' or 'context' in request body."}), 400

    text = data['text']
    context = data['context']

    # Step 1: Extract facts
    facts = extractor.extract(text)

    # Step 2: Verify facts
    verification = verifier.verify_response(facts, context)

    return jsonify({
        "input_text": text,
        "extracted_facts": facts,
        "verification": verification,
        "summary": verification.get("summary", {}),
    })


@app.route('/api/bias-check', methods=['POST'])
def bias_check():
    """
    Quick bias check on text using keyword-based analysis.
    
    Request body:
        {"text": "Text to analyze for bias"}
    
    Returns:
        Bias analysis results.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body."}), 400

    text = data['text']
    text_lower = text.lower()

    # Gender term analysis
    gender_terms = {
        'masculine': ['he', 'him', 'his', 'man', 'boy', 'male', 'king', 'prince'],
        'feminine': ['she', 'her', 'hers', 'woman', 'girl', 'female', 'queen', 'princess'],
        'neutral': ['they', 'them', 'theirs', 'person', 'human', 'individual'],
    }

    scores = {}
    for gender, terms in gender_terms.items():
        count = sum(text_lower.count(f" {t} ") + text_lower.count(f" {t}.") + text_lower.count(f" {t},") for t in terms)
        scores[gender] = count

    total = sum(scores.values())
    normalized = {k: round(v / total, 3) if total > 0 else 0 for k, v in scores.items()}

    # Stereotype check
    stereotypes = [
        'women are emotional', 'men are strong', 'asians are good at math',
        'women like cooking', 'men are breadwinners',
    ]
    has_stereotypes = any(phrase in text_lower for phrase in stereotypes)

    return jsonify({
        "gender_bias_scores": normalized,
        "raw_counts": scores,
        "has_stereotypes": has_stereotypes,
        "text_length_words": len(text.split()),
    })


# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting XAI API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
