from transformers import AutoModelForCausalLM
import os

MODEL_ID = os.environ["BF16_MODEL"]

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map={"": "cpu"},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

interesting = []
linear_count = 0

for name, module in model.named_modules():
    cls_name = module.__class__.__name__
    cls = cls_name.lower()
    lname = name.lower()

    if cls_name == "Linear":
        linear_count += 1

    if any(x in lname or x in cls for x in [
        "router",
        "gate",
        "expert",
        "moe",
        "lm_head",
        "norm",
        "deltanet",
        "delta",
        "linear_attn",
        "linear_attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]):
        interesting.append((name, cls_name))

print("Total Linear modules:", linear_count)

print("\nInteresting modules:")
for name, cls_name in interesting[:3000]:
    print(f"{name} :: {cls_name}")

print("\nLast 100 interesting modules:")
for name, cls_name in interesting[-100:]:
    print(f"{name} :: {cls_name}")
