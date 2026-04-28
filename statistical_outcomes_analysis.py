import json
import os

def display_statistical_outcomes(filepath="statistical_outcomes.json"):
    """
    Reads the statistical outcomes JSON file and displays a formatted summary.
    """
    if not os.path.exists(filepath):
        print(f"Error: Could not find '{filepath}'.")
        return

    with open(filepath, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"Error: '{filepath}' contains invalid JSON.")
            return

    metrics = data.get("evaluation_metrics", {})
    hallucination = metrics.get("hallucination_detection", {})
    bias = metrics.get("bias_detection", {})
    system = metrics.get("system_performance", {})

    print("\n" + "="*60)
    print(" [📊] STATISTICAL PERFORMANCE SUMMARY")
    print("="*60)
    
    # Hallucination Metrics
    print(" 🔍 HALLUCINATION DETECTION:")
    print(f"      Accuracy:               {hallucination.get('accuracy', 0) * 100:.1f}%")
    print(f"      Precision:              {hallucination.get('precision', 0) * 100:.1f}%")
    print(f"      Recall:                 {hallucination.get('recall', 0) * 100:.1f}%")
    print(f"      Mean Confidence:        {hallucination.get('mean_confidence', 0) * 100:.1f}%")
    print(f"      Dataset Evaluated:      {hallucination.get('benchmark_dataset', 'N/A')} ({hallucination.get('samples_evaluated', 0)} samples)")
    print("-" * 60)

    # Bias Detection Metrics
    print(" ⚖️  BIAS DETECTION:")
    print(f"      Stereotype Hit Rate:    {bias.get('stereotype_hit_rate', 0) * 100:.1f}%")
    print(f"      Gender Sensitivity:     {bias.get('gender_sensitivity', 0) * 100:.1f}%")
    print(f"      False Positive Rate:    {bias.get('false_positive_rate', 0) * 100:.1f}%")
    print(f"      Dataset Evaluated:      {bias.get('benchmark_dataset', 'N/A')} ({bias.get('samples_evaluated', 0)} samples)")
    print("-" * 60)

    # System Performance
    print(" ⚡ SYSTEM PERFORMANCE:")
    latency = system.get('latency_ms', {})
    print(f"      Fact Extraction:        {latency.get('fact_extraction', 0)}ms")
    print(f"      Verification:           {latency.get('verification', 0)}ms")
    print(f"      Bias Audit:             {latency.get('bias_audit', 0)}ms")
    print(f"      Total Pipeline:         {latency.get('total_pipeline', 0)}ms")
    print(f"      Throughput:             {system.get('throughput', 'N/A')}")
    print(f"      Environment:            {system.get('environment', 'N/A')}")
    
    print("="*60)
    print(f" Last Run: {data.get('last_run', 'N/A')}")
    print("="*60 + "\n")

if __name__ == "__main__":
    display_statistical_outcomes()
