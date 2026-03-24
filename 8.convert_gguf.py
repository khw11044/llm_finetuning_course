import os
import shutil
import glob
from huggingface_hub import login, create_repo, upload_folder
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

hf_token = os.getenv("HF_TOKEN")


max_seq_length = 2048
dtype = None
load_in_4bit = True


model_repo = "AeiROBOT/EDIE_qwen2.5-0.5B-Instruct_EDIE_MIND_Emotion-merged"

# checkpoint-10000에서 모델 + LoRA 어댑터 로드
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_repo,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

print("모델 로드 완료")


model_repo = "AeiROBOT/EDIE_qwen2.5-0.5B-Instruct_EDIE_MIND_Emotion"

gguf_repo = model_repo + "-gguf"

quantization_methods = ["f16", "q8_0", "q5_k_m", "q4_k_m"]

# 1. GGUF 변환 및 로컬 저장
model.save_pretrained_gguf(
    gguf_repo,
    tokenizer,
    quantization_method=quantization_methods,
)

# 2. CWD에 생성된 .gguf 파일들을 gguf_repo 폴더로 이동
#    (Unsloth가 양자화 파일을 CWD에 생성하는 문제 대응)
import glob, shutil
for gguf_file in glob.glob("*.gguf"):
    dest = os.path.join(gguf_repo, gguf_file)
    shutil.move(gguf_file, dest)
    print(f"[moved] {gguf_file} -> {dest}")

# Modelfile도 이동
if os.path.exists("Modelfile"):
    shutil.move("Modelfile", os.path.join(gguf_repo, "Modelfile"))

# 3. 모델 설정 및 토크나이저 파일들을 해당 폴더에 추가 저장
model.save_pretrained(gguf_repo)
tokenizer.save_pretrained(gguf_repo)

# 4. 확인
print("\n--- gguf_repo 폴더 내 파일 ---")
for f in sorted(os.listdir(gguf_repo)):
    size_mb = os.path.getsize(os.path.join(gguf_repo, f)) / 1024 / 1024
    print(f"  {f} ({size_mb:.1f} MB)")

# 5. 허깅페이스 리포 생성
create_repo(repo_id=gguf_repo, token=hf_token, private=True, exist_ok=True)

# 6. 폴더 전체 업로드 (.gguf 파일 + 모든 json 파일들)
upload_folder(
    repo_id=gguf_repo,
    folder_path=gguf_repo,
    token=hf_token,
)

print(f"\nGGUF 모델 및 설정 파일 업로드 완료: https://huggingface.co/{gguf_repo}")