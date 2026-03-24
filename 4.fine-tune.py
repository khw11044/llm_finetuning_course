import os
import argparse
import shutil
import glob

# ────────────────────────────────────────────────────────────
# 인자 파싱
# ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="EDIE Fine-tuning with Unsloth + LoRA")

# GPU
parser.add_argument("--gpu", type=str, default="1", help="CUDA_VISIBLE_DEVICES (default: 1)")

# 모델
parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--load_in_4bit", action="store_true", default=True)
parser.add_argument("--no_4bit", action="store_true", help="Disable 4bit quantization")

# 데이터셋
parser.add_argument("--dataset_repo", type=str, default="AeiROBOT/EDIE-emotion-mind-dataset")

# LoRA
parser.add_argument("--rank_r", type=int, default=32, help="LoRA rank (8, 16, 32, 64 권장)")
parser.add_argument("--lora_dropout", type=float, default=0.1)

# 학습
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=50)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine"])
parser.add_argument("--warmup_steps", type=int, default=5)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--dataset_num_proc", type=int, default=32)
parser.add_argument("--seed", type=int, default=3407)

# 저장 / 업로드
parser.add_argument("--model_repo", type=str, default="AeiROBOT/EDIE_qwen2.5-0.5B-Instruct_EDIE_MIND_Emotion")
parser.add_argument("--quantization_methods", type=str, nargs="+", default=["f16", "q8_0", "q5_k_m", "q4_k_m"])
parser.add_argument("--skip_upload", action="store_true", help="로컬 저장만 하고 HF 업로드 스킵")

args = parser.parse_args()

if args.no_4bit:
    args.load_in_4bit = False

# ────────────────────────────────────────────────────────────
# GPU 설정
# ────────────────────────────────────────────────────────────
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import numpy as np
from transformers import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

# ────────────────────────────────────────────────────────────
# 환경변수 & 시드
# ────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

np.random.seed(42)
set_seed(42)

# ────────────────────────────────────────────────────────────
# 모델 로드
# ────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=args.max_seq_length,
    dtype=None,
    load_in_4bit=args.load_in_4bit,
)

# chat_template 적용
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# ────────────────────────────────────────────────────────────
# 데이터셋 로드
# ────────────────────────────────────────────────────────────
from datasets import load_dataset

train_dataset = load_dataset(args.dataset_repo, split="train")
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

eval_dataset = load_dataset(args.dataset_repo, split="test")
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

# ────────────────────────────────────────────────────────────
# LoRA 설정
# ────────────────────────────────────────────────────────────
lora_alpha_scale = args.rank_r * 2

model = FastLanguageModel.get_peft_model(
    model,
    r=args.rank_r,
    lora_alpha=lora_alpha_scale,
    lora_dropout=args.lora_dropout,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=args.seed,
    use_rslora=False,
    loftq_config=None,
)

# ────────────────────────────────────────────────────────────
# 학습
# ────────────────────────────────────────────────────────────
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    args=SFTConfig(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        output_dir="outputs",
        report_to="none",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 학습 결과 출력
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ────────────────────────────────────────────────────────────
# 저장 및 업로드
# ────────────────────────────────────────────────────────────
from huggingface_hub import create_repo, upload_folder

hf_token = os.getenv("HF_TOKEN")
model_repo = args.model_repo
script_dir = os.path.dirname(os.path.abspath(__file__))

# [1/3] Merged 모델 저장 (★ 반드시 가장 먼저!)
merged_model_repo = model_repo + "-merged"

model.save_pretrained_merged(
    merged_model_repo,
    tokenizer,
    save_method="merged_16bit",
)

if not args.skip_upload:
    create_repo(repo_id=merged_model_repo, token=hf_token, private=True, exist_ok=True)
    upload_folder(repo_id=merged_model_repo, folder_path=merged_model_repo, token=hf_token)
print(f"[1/3] 병합 모델 완료: {merged_model_repo}")

# [2/3] GGUF 변환
gguf_repo = model_repo + "-gguf"

model.save_pretrained_gguf(
    gguf_repo,
    tokenizer,
    quantization_method=args.quantization_methods,
)

# unsloth가 script_dir에 .gguf / Modelfile을 생성할 수 있으므로 gguf_repo로 이동
os.makedirs(gguf_repo, exist_ok=True)

for gguf_file in glob.glob(os.path.join(script_dir, "*.gguf")):
    dest = os.path.join(gguf_repo, os.path.basename(gguf_file))
    shutil.move(gguf_file, dest)
    print(f"Moved: {os.path.basename(gguf_file)} -> {gguf_repo}/")

modelfile_path = os.path.join(script_dir, "Modelfile")
if os.path.exists(modelfile_path):
    shutil.move(modelfile_path, os.path.join(gguf_repo, "Modelfile"))

tokenizer.save_pretrained(gguf_repo)

if not args.skip_upload:
    create_repo(repo_id=gguf_repo, token=hf_token, private=True, exist_ok=True)
    upload_folder(repo_id=gguf_repo, folder_path=gguf_repo, token=hf_token)
print(f"[2/3] GGUF 모델 완료: {gguf_repo}")



# [3/3] LoRA adapter 저장
model.save_pretrained(model_repo)
tokenizer.save_pretrained(model_repo)

if not args.skip_upload:
    create_repo(repo_id=model_repo, token=hf_token, private=True, exist_ok=True)
    upload_folder(repo_id=model_repo, folder_path=model_repo, token=hf_token)
print(f"[3/3] LoRA adapter 완료: {model_repo}")

print("\n모든 저장 및 업로드 완료!")
