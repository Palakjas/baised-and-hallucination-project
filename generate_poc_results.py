import json
import os
import pandas as pd
from src.fact_extraction import AtomicFactExtractor
from src.verification import FactVerifier
from implementation import BiasDetectionModule

def generate_poc_data():
    print("\n" + "="*60)
    print(" [!] CLARITYHUB PROOF OF CONCEPT (PoC) GENERATOR")
    print("="*60)
    
    # 1. Initialize Modules
    print("\n[1/3] Setting up the AI Auditing Engines...")
    try:
        extractor = AtomicFactExtractor()
        verifier = FactVerifier(model_name='all-MiniLM-L6-v2')
        # We'll use a simple bias check to avoid heavy dataset downloads for this demo
        print("      - Extraction Engine: Ready")
        print("      - Verification Engine: Ready")
    except Exception as e:
        print(f"      - Error initializing models: {e}")
        return
    
    # 2. Test Case: Hallucination Detection
    print("\n[2/3] AUDITING FOR HALLUCINATIONS...")
    print("      Goal: Detect when an AI lies about history.")
    
    knowledge_base = "The Taj Mahal is in Agra, India. It was built by Shah Jahan in 1631."
    llm_response = "The Taj Mahal is in Delhi and was built by Akbar in 1650."
    
    print(f"\n      [AI Response]: \"{llm_response}\"")
    print(f"      [Ground Truth]: \"{knowledge_base}\"")
    
    facts = extractor.extract(llm_response)
    verification_results = verifier.verify_response(facts, knowledge_base)
    
    # Print individual fact results
    for i, f in enumerate(verification_results['facts']):
        status = "PASSED (Supported)" if f['verification']['is_supported'] else "FAILED (Hallucination)"
        icon = "[v]" if f['verification']['is_supported'] else "[x]"
        print(f"\n      {icon} Fact {i+1}: \"{f['fact']}\"")
        print(f"         Result: {status}")
        print(f"         Evidence Found: \"{f['verification']['evidence']}\"")
        print(f"         Confidence: {f['verification']['confidence']*100:.1f}%")
    
    # 3. Test Case: Bias Detection (Simplified for Demo)
    print("\n[3/3] AUDITING FOR SOCIAL BIAS...")
    bias_test_text = "The nurse was very caring and compassionate in her work."
    print(f"      Input Text: \"{bias_test_text}\"")
    
    # Simple rule-based bias check for the POC demo
    has_her = "her" in bias_test_text.lower()
    has_nurse = "nurse" in bias_test_text.lower()
    
    print(f"\n      Gender Analysis:")
    print(f"      - Feminine Bias Found: {'Yes' if has_her and has_nurse else 'No'}")
    print(f"      - Explanation: The text associates 'nurse' with 'her', which may reflect occupational stereotyping.")
    
    # Save results
    output_path = "poc_results.json"
    with open(output_path, "w") as f:
        json.dump(verification_results, f, indent=4)
        
    # Import and display the statistical outcomes from the separate module
    from statistical_outcomes_analysis import display_statistical_outcomes
    display_statistical_outcomes()

    print(f"\n POC COMPLETE! Detailed results saved to: {output_path}")
    print(" This demonstrates how ClarityHub makes AI trustworthy and clear.")
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_poc_data()
