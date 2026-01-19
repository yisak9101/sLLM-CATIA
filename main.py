import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from transformers import EarlyStoppingCallback
# -----------------------
# 0) Config
# -----------------------
MODEL_NAME = "kakaocorp/kanana-1.5-2.1b-instruct-2505"

# 로컬 파일 경로 (예: train.jsonl / valid.jsonl)
TRAIN_FILE = "train.jsonl"
VALID_FILE = None  # 없으면 train에서 자동 분리

MAX_LEN = 512

# JSON만 출력하도록 강하게 유도하는 시스템 프롬프트
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
    # Instruct 스타일(일관된 템플릿)
    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### User:\n{user_prompt.strip()}\n\n"
        f"### Assistant:\n"
    )

# -----------------------
# 1) Tokenizer / Model
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
dtype = torch.bfloat16 if use_bf16 else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
)

# -----------------------
# 2) LoRA 설정
# -----------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------
# 3) Dataset Load (JSONL)
# -----------------------
data_files = {"train": TRAIN_FILE}
if VALID_FILE:
    data_files["validation"] = VALID_FILE

ds = load_dataset("json", data_files=data_files)

if "validation" not in ds:
    split = ds["train"].train_test_split(test_size=0.05, seed=42)
    train_ds, valid_ds = split["train"], split["test"]
else:
    train_ds, valid_ds = ds["train"], ds["validation"]

# -----------------------
# 4) Preprocess: completion(dict) -> JSON string, prompt template 적용
# -----------------------
def normalize_example(ex):
    # completion이 dict이면 JSON 문자열로 변환 (키 순서 고정 + 공백 최소화)
    comp = ex["completion"]
    if isinstance(comp, dict):
        comp_str = json.dumps(comp, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        # 이미 문자열이면 그대로 사용(사용자가 직접 JSON string 만들어둔 경우)
        comp_str = str(comp).strip()

    prompt = build_prompt(ex["prompt"])
    return {"prompt_text": prompt, "completion_text": comp_str}

train_ds = train_ds.map(normalize_example, remove_columns=train_ds.column_names)
valid_ds = valid_ds.map(normalize_example, remove_columns=valid_ds.column_names)

# -----------------------
# 5) Tokenize + labels 마스킹
#    - prompt 부분은 -100
#    - 정답(JSON) 부분만 loss 반영
# -----------------------
def tokenize_and_mask(ex):
    prompt = ex["prompt_text"]
    completion = ex["completion_text"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    # JSON 정답 뒤에 EOS를 붙여 "여기서 끝"을 학습
    comp_ids = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + comp_ids)[:MAX_LEN]
    labels = ([-100] * len(prompt_ids) + comp_ids)[:MAX_LEN]
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_tok = train_ds.map(tokenize_and_mask, remove_columns=train_ds.column_names)
valid_tok = valid_ds.map(tokenize_and_mask, remove_columns=valid_ds.column_names)

# -----------------------
# 6) Collator (패딩 + label 패딩 -100 처리)
# -----------------------
def collate_fn(features):
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    if "labels" in batch:
        batch["labels"] = batch["labels"].masked_fill(batch["labels"] == tokenizer.pad_token_id, -100)
    return batch

# -----------------------
# 7) Train
# -----------------------
args = TrainingArguments(
    output_dir="./kanana_case_json_lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="steps",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    bf16=use_bf16,
    fp16=not use_bf16,
    report_to="none",
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# LoRA 어댑터 저장
out_dir = "./kanana_case_json_lora_adapter_revised"
trainer.save_model(out_dir)
tokenizer.save_pretrained(out_dir)
print(f"Done. Saved to {out_dir}")