#!/usr/bin/env python3
"""
문장에서 특수문자를 제거한 버전을 추가하여 데이터 증강하는 스크립트

사용법:
    python 1.augment_dataset.py --data data_raw/ANGRY.txt --output data_aug
"""
import argparse
import re
from pathlib import Path


def augment_lines(lines: list[str]) -> list[str]:
    """각 문장에 대해 특수문자가 있으면 제거한 버전을 추가"""
    # 제거할 특수문자 패턴
    special_chars = r'[.,!?~…·\-_\'\";:()（）「」『』【】\[\]<>《》]'

    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 원본 문장 추가
        result.append(line)

        # 특수문자가 있는지 확인
        if re.search(special_chars, line):
            # 특수문자 제거한 버전 추가
            cleaned = re.sub(special_chars, '', line)
            # 연속 공백 정리
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned and cleaned != line:
                result.append(cleaned)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="문장에서 특수문자를 제거한 버전을 추가하여 데이터 증강"
    )
    parser.add_argument("--data", required=True, help="입력 텍스트 파일 경로")
    parser.add_argument("--output", required=True, help="출력 디렉토리 경로")
    args = parser.parse_args()

    input_path = Path(args.data)
    output_dir = Path(args.output)

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 파일 읽기
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"[info] 입력 파일: {input_path}")
    print(f"[info] 원본 문장 수: {len([l for l in lines if l.strip()])}")

    # 증강
    augmented = augment_lines(lines)

    print(f"[info] 증강 후 문장 수: {len(augmented)}")

    # 출력 파일 저장 (같은 이름으로)
    output_path = f"./{output_dir}/{input_path.name}"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in augmented:
            f.write(line + "\n")

    print(f"[done] 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
