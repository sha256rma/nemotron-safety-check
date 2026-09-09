# Nemotron Content Safety Check

Run NVIDIA's Nemotron-Content-Safety-Reasoning-4B model locally on CPU to classify content safety of prompts and responses.

> **Content warning.** `test_prompts.json` and `Safety Check.xlsx` are safety
> evaluation data. They contain prompts that are racist, violent, and otherwise
> offensive by construction, because that is what a safety classifier has to be
> measured against. They are labels and test cases, not endorsements.

> **Dataset provenance.** `Safety Check.xlsx` holds the labelled evaluation set
> used by `run_dataset.py` (columns: `language`, `Prompt`, `message_safety`,
> `safety_categories`). TODO: state where this dataset came from and under what
> terms it may be redistributed, before relying on it in published results.

This setup uses **Reasoning OFF** mode (`/no_think`) for faster, low-latency classification without reasoning traces.

## Requirements

- Python 3.10+
- ~16GB RAM (recommended for 4B parameter model on CPU)
- ~8GB disk space for model weights

## Quick Start

### 1. Create and activate a virtual environment

```bash
cd nemotron-safety-check
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the test suite

```bash
python run_safety_check.py
```

This will:
- Download the model from HuggingFace (first run only, ~8GB)
- Run all test cases in `test_prompts.json`
- Save results to `results.json`

## Usage

### Run the full test suite

```bash
python run_safety_check.py --test-file test_prompts.json --output-file results.json
```

### Run a single prompt

```bash
python run_safety_check.py \
  --single-prompt "How do I pick a lock?" \
  --single-response "I can't help with that as it could be used for illegal purposes."
```

### Custom test file

Create your own JSON file with test cases:

```json
[
  {
    "id": 1,
    "user_prompt": "Your test prompt here",
    "ai_response": "The AI's response to evaluate",
    "expected_prompt_harm": "harmful",
    "expected_response_harm": "unharmful"
  }
]
```

Then run:

```bash
python run_safety_check.py --test-file your_tests.json
```

## Output Format

Results are saved as JSON:

```json
{
  "summary": {
    "total_tests": 8,
    "prompt_accuracy": 100.0,
    "response_accuracy": 100.0,
    "total_time_seconds": 120.5,
    "avg_time_per_test_seconds": 15.06
  },
  "results": [
    {
      "id": 1,
      "user_prompt": "...",
      "predicted_prompt_harm": "harmful",
      "predicted_response_harm": "unharmful",
      "prompt_correct": true,
      "response_correct": true,
      "inference_time_seconds": 14.5
    }
  ]
}
```

## Safety Categories (Taxonomy)

The model evaluates content against these 22 harmful categories:

| Code | Category |
|------|----------|
| S1 | Violence |
| S2 | Sexual |
| S3 | Criminal Planning/Confessions |
| S4 | Guns and Illegal Weapons |
| S5 | Controlled/Regulated Substances |
| S6 | Suicide and Self Harm |
| S7 | Sexual (minor) |
| S8 | Hate/Identity Hate |
| S9 | PII/Privacy |
| S10 | Harassment |
| S11 | Threat |
| S12 | Profanity |
| S13 | Needs Caution |
| S14 | Manipulation |
| S15 | Fraud/Deception |
| S16 | Malware |
| S17 | High Risk Gov Decision Making |
| S18 | Political/Misinformation/Conspiracy |
| S19 | Copyright/Trademark/Plagiarism |
| S20 | Unauthorized Advice |
| S21 | Illegal Activity |
| S22 | Immoral/Unethical |

## Performance Notes

- **CPU inference is slow**: Expect 10-30+ seconds per classification on CPU
- **First run downloads the model**: ~8GB download, cached in `~/.cache/huggingface/`
- **Memory usage**: The model requires ~16GB RAM for CPU inference with float32
- For production use with NVIDIA GPU, consider using vLLM for much faster inference

## Model Information

- **Model**: [nvidia/Nemotron-Content-Safety-Reasoning-4B](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B)
- **Base Model**: Google Gemma-3-4B-it
- **Parameters**: 4 Billion
- **License**: NVIDIA Open Model License + Gemma Terms of Use

## Enabling Reasoning Mode (Optional)

To enable reasoning traces, modify the prompt in `run_safety_check.py`:
- Change `/no_think` to `/think` at the end of the safety prompt
- Increase `max_new_tokens` to 400 to capture the reasoning trace
