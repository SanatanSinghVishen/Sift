import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "SanatanSinghVishen/sift-1b-dpo")
model.eval()

# Load 3 holdout rows from the end of sft_dataset
rows = []
with open("data/sft_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

samples = rows[-3:]

for idx, sample in enumerate(samples):
    convs = sample["conversations"]
    prompt = [c for c in convs if c["role"] in ("system", "user")]
    expected = [c["content"] for c in convs if c["role"] == "assistant"][0]
    
    encoded = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) or hasattr(encoded, "input_ids") else encoded
    outputs = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    
    print(f"\n--- SAMPLE {idx+1} ---")
    print("USER:", prompt[-1]["content"])
    print("EXPECTED :", expected)
    print("GENERATED:", generated)
    print("MATCH?", generated == expected)
