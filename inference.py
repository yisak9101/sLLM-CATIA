import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
adapter = "./kanana_case_json_lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(base)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter)

SYSTEM_PROMPT = (
    "너는 케이스(박스) 치수를 설계하는 도우미다.\n"
    "사용자 요청을 읽고 height, width, depth를 밀리미터(mm) 정수로 결정해라.\n"
    "반드시 아래 JSON 형식만 출력하고, 다른 설명/텍스트는 출력하지 마라.\n"
    '형식: {"height": <int>, "width": <int>, "depth": <int>}\n'
)

def build_prompt(user_prompt: str) -> str:
    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### User:\n{user_prompt.strip()}\n\n"
        f"### Assistant:\n"
    )

prompt = build_prompt("높이는 낮고 가로는 넓게, 깊이는 중간 정도로 해줘")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
text = tokenizer.decode(out[0], skip_special_tokens=True)

print(text)