#!/usr/bin/env python3
"""
CSV 파일들을 통합하여 Hugging Face 데이터셋 레파지토리에 업로드하는 스크립트

python 3.merge_upload.py

"""
#!/usr/bin/env python3
"""
ChatML 형식의 CSV 파일들을 통합하여 Hugging Face 데이터셋 레파지토리에 업로드하는 스크립트
설정된 REPO_ID로 train/test split이 적용된 데이터셋이 업로드됩니다.

실행 전: export HUGGINGFACE_HUB_TOKEN=your_token_here
"""
import os
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
# .env 로드
current_directory = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_directory, '.env')
load_dotenv(env_path)

# ===== 사용자 설정 =====
REPO_ID = "AeiROBOT/EDIE-emotion-mind-dataset"
DATA_DIR = Path(__file__).parent / "data"
README_PATH = Path(__file__).parent / "README.md"

def main():
    # 1) 토큰 가져오기
    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError(
            "HUGGINGFACE_HUB_TOKEN 환경변수가 설정되어 있지 않습니다. "
            "https://huggingface.co/settings/tokens 에서 토큰을 만드세요."
        )

    # 2) 모든 CSV 파일 통합
    # 파일명 패턴이 *_chatml_dataset.csv 인 것만 찾아 통합합니다.
    csv_files = list(DATA_DIR.glob("*_chatml_dataset.csv"))
    if not csv_files:
        # 패턴에 맞는 파일이 없으면 모든 .csv 시도
        csv_files = list(DATA_DIR.glob("*.csv"))

    print(f"[info] 발견된 CSV 파일: {len(csv_files)}개")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"  - {csv_file.name}: {len(df)}행")
        dfs.append(df)

    if not dfs:
        print("[error] 통합할 데이터가 없습니다.")
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    
    # [핵심] CSV의 텍스트 형태 'messages' 컬럼을 파이썬 리스트/딕셔너리 객체로 변환
    print("[info] JSON 문자열을 데이터셋 객체로 변환 중...")
    try:
        merged_df['messages'] = merged_df['messages'].apply(json.loads)
    except Exception as e:
        print(f"[error] JSON 파싱 중 오류 발생: {e}")
        print("데이터셋 형식이 ChatML(messages 컬럼)이 맞는지 확인하세요.")
        return

    print(f"[info] 통합된 총 행 수: {len(merged_df)}")

    # 3) Dataset 객체 생성 (ChatML은 'messages' 컬럼 하나만 사용)
    dataset = Dataset.from_pandas(merged_df[["messages"]])

    # 4) train/test 분할 (90% train, 10% test)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "test": split_dataset["test"],
    })
    
    print(f"[info] Dataset 구조:")
    print(f"  - train: {len(dataset_dict['train'])}행")
    print(f"  - test: {len(dataset_dict['test'])}행")

    # 5) 리포지터리 생성
    create_repo(
        repo_id=REPO_ID,
        token=token,
        private=False,
        exist_ok=True,
        repo_type="dataset",
    )
    print(f"[info] 리포지터리 생성/확인 완료: {REPO_ID}")

    # 6) 기존 데이터 파일 삭제 후 새로 업로드 (선택 사항)
    api = HfApi()
    try:
        api.delete_folder(
            repo_id=REPO_ID,
            path_in_repo="data",
            token=token,
            repo_type="dataset",
        )
        print("[info] 기존 data 폴더 삭제 완료")
    except Exception:
        pass 

    # 7) 데이터셋 업로드
    dataset_dict.push_to_hub(
        repo_id=REPO_ID,
        token=token,
        private=True,
    )
    print("[info] 데이터셋 업로드 완료")

    # 8) README 업로드 (존재할 경우만)
    if README_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(README_PATH),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            token=token,
            repo_type="dataset",
        )
        print("[info] README.md 업로드 완료")

    print(f"[done] 업로드 완료: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()