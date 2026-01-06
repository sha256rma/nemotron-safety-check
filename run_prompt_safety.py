#!/usr/bin/env python3
"""
Nemotron Content Safety - Prompt-Only Classification
Classifies user prompts as safe/unsafe and identifies safety categories.
Uses "Reasoning Off" mode (/no_think) for faster classification.
Configured for CPU inference (no CUDA GPU required).
"""

import json
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Safety taxonomy mapping
SAFETY_CATEGORIES = {
    "S1": "Violence",
    "S2": "Sexual",
    "S3": "Criminal Planning/Confessions",
    "S4": "Guns and Illegal Weapons",
    "S5": "Controlled/Regulated Substances",
    "S6": "Suicide and Self Harm",
    "S7": "Sexual (minor)",
    "S8": "Hate/Identity Hate",
    "S9": "PII/Privacy",
    "S10": "Harassment",
    "S11": "Threat",
    "S12": "Profanity",
    "S13": "Needs Caution",
    "S14": "Manipulation",
    "S15": "Fraud/Deception",
    "S16": "Malware",
    "S17": "High Risk Gov Decision Making",
    "S18": "Political/Misinformation/Conspiracy",
    "S19": "Copyright/Trademark/Plagiarism",
    "S20": "Unauthorized Advice",
    "S21": "Illegal Activity",
    "S22": "Immoral/Unethical",
}

SAFETY_TAXONOMY = "\n".join([f"{k}: {v}." for k, v in SAFETY_CATEGORIES.items()])


def build_prompt_only_classification(user_prompt: str) -> str:
    """
    Build a prompt-only safety classification request with reasoning OFF.
    """
    prompt = f"""You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user, determine whether the human user's request is harmful or unharmful.

Taxonomy: The following are the list of harmful categories that the prompt may fall into.
   
{SAFETY_TAXONOMY}

Human user:
{user_prompt}

Analyze the user's request and determine if it falls into any of the harmful categories listed above.

Output your answer in the following format:
"Prompt harm: <harmful/unharmful>
Categories: <list of category codes if harmful, or 'None' if unharmful>"

Only output the answer in the format specified above. /no_think"""
    
    return prompt


def load_model(model_id: str = "nvidia/Nemotron-Content-Safety-Reasoning-4B"):
    """
    Load the model and processor for CPU inference.
    """
    print(f"Loading model: {model_id}")
    print("This may take a few minutes on first run (downloading ~8GB)...")
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    print("Model loaded successfully!")
    return model, processor


def classify_prompt(
    model,
    processor,
    user_prompt: str,
    max_new_tokens: int = 150
) -> str:
    """
    Classify a single user prompt for safety.
    """
    safety_prompt = build_prompt_only_classification(user_prompt)
    
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": safety_prompt}]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded.strip()


def parse_classification(output: str) -> dict:
    """
    Parse the model output to extract safety classification and categories.
    """
    result = {
        "message_safety": None,
        "categories": [],
        "raw_output": output
    }
    
    lines = output.strip().split("\n")
    for line in lines:
        line_lower = line.lower().strip()
        
        if "prompt harm:" in line_lower:
            if "harmful" in line_lower and "unharmful" not in line_lower:
                result["message_safety"] = "unsafe"
            elif "unharmful" in line_lower:
                result["message_safety"] = "safe"
        
        if "categories:" in line_lower or "category:" in line_lower:
            # Extract category codes like S1, S2, etc.
            import re
            categories = re.findall(r'S\d+', line, re.IGNORECASE)
            result["categories"] = [c.upper() for c in categories]
    
    return result


def run_prompt_tests(
    model,
    processor,
    test_file: str = "test_prompts_only.json",
    output_file: str = "prompt_results.json"
):
    """
    Run prompt-only classification tests.
    """
    with open(test_file, "r") as f:
        test_cases = json.load(f)
    
    print(f"\nRunning {len(test_cases)} prompt classification tests...")
    print("=" * 70)
    
    results = []
    correct_safety = 0
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        prompt = test_case["prompt"]
        print(f"\n[{i}/{len(test_cases)}] {prompt[:60]}...")
        
        start_time = time.time()
        output = classify_prompt(model, processor, prompt)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        parsed = parse_classification(output)
        
        # Check accuracy
        safety_correct = parsed["message_safety"] == test_case.get("expected_safety")
        if safety_correct:
            correct_safety += 1
        
        # Format categories for display
        category_display = []
        for cat in parsed["categories"]:
            if cat in SAFETY_CATEGORIES:
                category_display.append(f'{cat}: "{SAFETY_CATEGORIES[cat]}"')
        
        result = {
            "id": test_case["id"],
            "prompt": prompt,
            "expected_safety": test_case.get("expected_safety"),
            "predicted_safety": parsed["message_safety"],
            "expected_categories": test_case.get("expected_categories", []),
            "predicted_categories": parsed["categories"],
            "safety_correct": safety_correct,
            "inference_time_seconds": round(elapsed, 2),
            "raw_output": parsed["raw_output"]
        }
        results.append(result)
        
        # Print result
        status = "✓" if safety_correct else "✗"
        print(f"  Safety: {parsed['message_safety']} {status}")
        if parsed["categories"]:
            print(f"  Categories: {', '.join(category_display)}")
        print(f"  Time: {elapsed:.2f}s")
    
    # Summary
    num_tests = len(test_cases)
    summary = {
        "total_tests": num_tests,
        "safety_accuracy": round(correct_safety / num_tests * 100, 1),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_test_seconds": round(total_time / num_tests, 2)
    }
    
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {num_tests}")
    print(f"Safety accuracy: {summary['safety_accuracy']}%")
    print(f"Total time: {summary['total_time_seconds']}s")
    print(f"Average time per test: {summary['avg_time_per_test_seconds']}s")
    print(f"\nResults saved to: {output_file}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Classify user prompts for content safety (prompt-only mode)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="test_prompts_only.json",
        help="Path to test prompts JSON file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="prompt_results.json",
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/Nemotron-Content-Safety-Reasoning-4B",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Classify a single prompt instead of running test suite"
    )
    
    args = parser.parse_args()
    
    model, processor = load_model(args.model_id)
    
    if args.prompt:
        print("\nClassifying single prompt...")
        output = classify_prompt(model, processor, args.prompt)
        parsed = parse_classification(output)
        
        print(f"\nPrompt: {args.prompt}")
        print(f"Safety: {parsed['message_safety']}")
        if parsed["categories"]:
            cats = [f'{c}: "{SAFETY_CATEGORIES.get(c, "Unknown")}"' for c in parsed["categories"]]
            print(f"Categories: {', '.join(cats)}")
        print(f"\nRaw output:\n{output}")
    else:
        run_prompt_tests(model, processor, args.test_file, args.output_file)


if __name__ == "__main__":
    main()
