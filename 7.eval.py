import os
import argparse


"""
python 7.eval.py --model_path ./AeiROBOT/EDIE_qwen2.5-0.5B-Instruct_EDIE_MIND_Emotion-merged --dataset_repo AeiROBOT/EDIE-emotion-mind-dataset



"""


parser = argparse.ArgumentParser(description="Fine-tuned 모델 데이터셋 평가")
parser.add_argument("--model_path", type=str, default="AeiROBOT/EDIE_qwen2.5-0.5B-Instruct_EDIE_MIND_Emotion-merged",
                    help="로컬 경로 또는 HuggingFace repo ID")
parser.add_argument("--dataset_repo", type=str, default="AeiROBOT/EDIE-emotion-mind-dataset")
parser.add_argument("--cuda_device", type=str, default="1")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--load_in_4bit", action="store_true", default=True)
parser.add_argument("--no_4bit", action="store_true", help="4bit 양자화 비활성화")
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
args = parser.parse_args()

if args.no_4bit:
    args.load_in_4bit = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import json
import re
import time
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import set_seed
from sklearn.metrics import accuracy_score, mean_absolute_error
from unsloth import FastLanguageModel

np.random.seed(42)
set_seed(42)
load_dotenv()

SYSTEM_PROMPT = "You are EDIE, a companion robot. Listen to the user's emotion-filled speech and output emotion, intensity, and ethogram."


def parse_json_response(text):
    if not text:
        return None
    text = text.strip()
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if all(k in parsed for k in ("emotion", "intensity", "ethogram")):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def generate_response(model, tokenizer, user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "input_ids"):
        input_ids = inputs["input_ids"].to("cuda")
    else:
        input_ids = inputs.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    response = tokenizer.batch_decode(outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)[0]
    return response.strip()


def intensity_bin(v):
    if v <= 0.3: return "weak"
    elif v <= 0.7: return "medium"
    else: return "strong"


def ethogram_category(e):
    return e.split("_")[0] if isinstance(e, str) and "_" in e else str(e)


if __name__ == "__main__":
    print(f"CUDA device: {args.cuda_device}")
    print(f"Available devices: {torch.cuda.device_count()}")

    # 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    # 데이터셋 로드 및 평가
    eval_dataset = load_dataset(args.dataset_repo, split="test")
    print(f"\n평가 데이터셋: {len(eval_dataset)}개 샘플")

    results = []
    parse_failures = 0
    total_time = 0

    for sample in tqdm(eval_dataset, desc="Evaluating"):
        msgs = sample["messages"]
        user_input = msgs[1]["content"]
        expected_raw = msgs[2]["content"]
        expected = parse_json_response(expected_raw)

        if expected is None:
            continue

        start_time = time.perf_counter()
        generated_raw = generate_response(model, tokenizer, user_input)
        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        predicted = parse_json_response(generated_raw)

        if predicted is None:
            parse_failures += 1

        results.append({
            "user_input": user_input,
            "expected": expected,
            "predicted": predicted,
            "generated_raw": generated_raw,
            "latency": elapsed,
        })

    print(f"\n평가 완료: {len(results)}개 샘플, JSON 파싱 실패: {parse_failures}건")
    print(f"평균 추론 시간: {total_time / len(results):.4f}s")

    # 파싱 실패 사례 출력
    fail_results = [r for r in results if r["predicted"] is None]
    print(f"\n{'='*60}")
    print(f"JSON 파싱 실패 사례 ({len(fail_results)}건)")
    print(f"{'='*60}")
    for i, r in enumerate(fail_results):
        print(f"\n[{i+1}/{len(fail_results)}]")
        print(f"  입력: {r['user_input'][:80]}")
        print(f"  정답: {r['expected']}")
        print(f"  LLM 출력: {repr(r['generated_raw'][:200])}")
        print(f"-"*60)

    # 메트릭 계산
    valid_results = [r for r in results if r["predicted"] is not None]

    expected_emotions = [r["expected"]["emotion"] for r in valid_results]
    predicted_emotions = [r["predicted"]["emotion"] for r in valid_results]
    emotion_accuracy = accuracy_score(expected_emotions, predicted_emotions)

    expected_intensities = [r["expected"]["intensity"] for r in valid_results]
    predicted_intensities = [r["predicted"]["intensity"] for r in valid_results]
    intensity_mae = mean_absolute_error(expected_intensities, predicted_intensities)
    intensity_bin_acc = accuracy_score(
        [intensity_bin(v) for v in expected_intensities],
        [intensity_bin(v) for v in predicted_intensities]
    )

    expected_ethograms = [r["expected"]["ethogram"] for r in valid_results]
    predicted_ethograms = [r["predicted"]["ethogram"] for r in valid_results]
    ethogram_accuracy = accuracy_score(expected_ethograms, predicted_ethograms)
    ethogram_cat_acc = accuracy_score(
        [ethogram_category(e) for e in expected_ethograms],
        [ethogram_category(e) for e in predicted_ethograms]
    )

    print("=" * 60)
    print(f"총 평가 샘플: {len(results)}")
    print(f"JSON 파싱 성공: {len(valid_results)} / {len(results)} ({len(valid_results)/len(results):.1%})")
    print(f"JSON 파싱 실패: {parse_failures}건")
    print("=" * 60)
    print(f"\n[Emotion]  정확도 (Exact Match): {emotion_accuracy:.2%}")
    print(f"[Intensity] MAE: {intensity_mae:.4f}")
    print(f"[Intensity] 구간 정확도 (weak/medium/strong): {intensity_bin_acc:.2%}")
    print(f"[Ethogram] 정확도 (Exact Match): {ethogram_accuracy:.2%}")
    print(f"[Ethogram] 대분류 정확도 (v/t/e/p/a/s): {ethogram_cat_acc:.2%}")
    print("=" * 60)
