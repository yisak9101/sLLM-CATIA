import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
adapter = "./kanana_case_json_lora_adapter_revised"

tokenizer = AutoTokenizer.from_pretrained(base)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter)

SYSTEM_PROMPT = (
    """
    너는 케이스(박스) 치수 설계 도우미다.
    입력은 항상 다음 형식이다:
    (현재 치수 height: H, width: W, depth: D) <요청문>
    요청에 따라 현재 치수를 참고할 수도 있고, 무시하고 새로 설계할 수도 있다.
    height, width, depth는 mm 단위 양의 정수로 결정한다.
    상대 표현(예: 줄여줘, 늘려줘, 조금, 많이, 더, 덜, 얇게, 두껍게, 좁게, 넓게) → 현재 치수 기반 변형
    절대 표현(예: 큰, 작은, 넓적한, 정사각형, 특정 수치 지정 등) → 새로 설계 가능
    출력은 설명 없이 JSON 한 줄만:
    {"height": <int>, "width": <int>, "depth": <int>}
    """
)

def build_prompt(user_prompt: str) -> str:
    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### User:\n{user_prompt.strip()}\n\n"
        f"### Assistant:\n"
    )

prompt = build_prompt("(현재 치수 height: 158, width: 137, depth: 70) 높이 좀 낮춰줘")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
text = tokenizer.decode(out[0], skip_special_tokens=True)

print(text)