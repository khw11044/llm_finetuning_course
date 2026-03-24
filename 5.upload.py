import os
import shutil
import glob
from huggingface_hub import login, create_repo, upload_folder

from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token

model_repo = "AeiROBOT/EDIE_qwen2.5-0.5B-Instruct_EDIE_MIND_Emotion"

# 저장 및 업로드

# 허깅페이스 리포 생성 및 업로드
create_repo(
    repo_id=model_repo, 
    token=hf_token, 
    private=True,
    exist_ok=True
)

upload_folder(
    repo_id=model_repo,
    folder_path=model_repo,
    token=hf_token,
)

print(f"병합 모델 업로드 완료: https://huggingface.co/{model_repo}")