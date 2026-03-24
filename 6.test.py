import os
import argparse


"""

while True + input("You> ") 루프
종료: q 입력 또는 Ctrl+C
tone 변경: /tone HAPPY 명령으로 대화 중 변경 가능

python 6.test.py --model_path ./AeiROBOT/EDIE_qwen2.5-0.5B-Instruct_EDIE_MIND_Emotion-merged --tone HAPPY --cuda_device 0



"""


parser = argparse.ArgumentParser(description="EDIE 모델 대화형 테스트")
parser.add_argument("--model_path", type=str, default="AeiROBOT/EDIE_qwen2.5-0.5B-Instruct_EDIE_MIND_Emotion-merged",
                    help="로컬 경로 또는 HuggingFace repo ID")
parser.add_argument("--cuda_device", type=str, default="1")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--load_in_4bit", action="store_true", default=True)
parser.add_argument("--no_4bit", action="store_true", help="4bit 양자화 비활성화")
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
parser.add_argument("--tone", type=str, default="NEUTRAL", help="기본 tone (대화 중 변경 가능)")
args = parser.parse_args()

if args.no_4bit:
    args.load_in_4bit = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import torch
from unsloth import FastLanguageModel

SYSTEM_PROMPT = "You are EDIE, a companion robot. Listen to the user's emotion-filled speech and output emotion, intensity, and ethogram."


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


if __name__ == "__main__":
    print(f"CUDA device: {args.cuda_device}")
    print(f"모델 로딩: {args.model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    tone = args.tone
    print(f"\n{'='*60}")
    print(f"EDIE 대화형 테스트 (tone: {tone})")
    print(f"  종료: q 입력 또는 Ctrl+C")
    print(f"  tone 변경: /tone HAPPY")
    print(f"{'='*60}\n")

    try:
        while True:
            text = input("You> ").strip()
            if not text:
                continue
            if text.lower() == "q":
                break
            if text.startswith("/tone "):
                tone = text.split(maxsplit=1)[1].strip()
                print(f"  [tone 변경: {tone}]\n")
                continue

            user_input = f"\ntone: {tone}\ntext: {text}\n"
            response = generate_response(model, tokenizer, user_input)
            print(f"EDIE> {response}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n")

    print("테스트 종료.")
