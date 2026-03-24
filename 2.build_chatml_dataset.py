import argparse
import os
import pandas as pd
import json
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# .env 로드
current_directory = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_directory, '.env')
load_dotenv(env_path)


"""
실행 예시:
python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/ANGRY.txt --output data
python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/ANGRY.txt --output data --model gpt-4.1


python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/ANGRY.txt --output data --model gpt-4o
python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/DISGUSTED.txt --output data --model gpt-4o
python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/FEARFUL.txt --output data --model gpt-4o
python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/HAPPY.txt --output data --model gpt-4o
python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/NEUTRAL.txt --output data --model gpt-4o
python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/SAD.txt --output data --model gpt-4o
python 2.build_chatml_dataset.py --prompt prompt/simple.txt --instruction data_aug/SURPRISED.txt --output data --model gpt-4o

"""



new_system_content="""
You are EDIE, a companion robot. Listen to the user's emotion-filled speech and output emotion, intensity, and ethogram.
"""


def get_llm(model_name="gpt-4o-mini", temperature=0.1):
    return ChatOpenAI(model_name=model_name, temperature=temperature)


def validate_response(response_text):
    """LLM 응답이 올바른 JSON 형식인지 검증한다."""
    try:
        # 이중 중괄호 {{ }} → 단일 { } 변환
        response_text = response_text.replace('{{', '{').replace('}}', '}')
        # 프롬프트 예시의 작은따옴표('a_03') → 큰따옴표("a_03") 변환
        response_text = response_text.replace("'", '"')
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        json_str = response_text[start:end]
        parsed = json.loads(json_str)

        if "emotion" in parsed and "intensity" in parsed and "ethogram" in parsed:
            return json_str
    except json.JSONDecodeError:
        pass
    return None


def build_chatml_dataset(prompt_path, instruction_path, output_path, model_name):
    # 1. 프롬프트 파일(System Prompt) 읽기
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_content = f.read().strip()
    except FileNotFoundError:
        print(f"오류: 프롬프트 파일을 찾을 수 없습니다. 경로: {prompt_path}")
        return

    # 2. 지시사항(User Input) 파일 읽기
    try:
        with open(instruction_path, 'r', encoding='utf-8') as f:
            instructions = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"오류: 지시사항 파일을 찾을 수 없습니다. 경로: {instruction_path}")
        return

    # 파일명에서 tone 추출 (예: ANGRY.txt -> ANGRY)
    file_name = os.path.basename(instruction_path)
    tone = os.path.splitext(file_name)[0].upper()

    # LLM 초기화
    llm = get_llm(model_name=model_name)

    # NEUTRAL이 아닌 감정이면 NEUTRAL 증강도 수행
    do_neutral_aug = (tone != "NEUTRAL")

    # 3. ChatML용 Messages 구조 생성 (LLM 호출)
    data = []
    failed = 0
    for instruction in tqdm(instructions, desc=f"Processing {tone}"):
        # === 원본 tone 호출 ===
        user_content = f"tone: {tone}\ntext: {instruction}"

        try:
            response = llm.invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ])
            raw_response = response.content
        except Exception as e:
            print(f"\nLLM 호출 오류: {e} (입력: {instruction[:30]}...)")
            failed += 1
            continue

        validated = validate_response(raw_response)
        if validated is None:
            print(f"\n응답 형식 오류 (입력: {instruction[:30]}...) 응답: {raw_response[:100]}")
            failed += 1
            continue

        messages = [
            {"role": "system", "content": new_system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": validated}
        ]
        data.append({
            "messages": json.dumps(messages, ensure_ascii=False)
        })

        # === NEUTRAL 증강 (LLM 호출 없이 intensity - 0.4) ===
        if do_neutral_aug:
            try:
                parsed = json.loads(validated)
                neutral_intensity = round(max(parsed["intensity"] - 0.4, 0.0), 2)
                neutral_result = json.dumps({
                    "emotion": parsed["emotion"],
                    "intensity": neutral_intensity,
                    "ethogram": parsed["ethogram"]
                }, ensure_ascii=False)

                neutral_user_content = f"tone: NEUTRAL\ntext: {instruction}"
                neutral_messages = [
                    {"role": "system", "content": new_system_content},
                    {"role": "user", "content": neutral_user_content},
                    {"role": "assistant", "content": neutral_result}
                ]
                data.append({
                    "messages": json.dumps(neutral_messages, ensure_ascii=False)
                })
            except (json.JSONDecodeError, KeyError):
                failed += 1

    # 4. 결과 저장
    os.makedirs(output_path, exist_ok=True)

    # 원본 감정 데이터셋 저장
    if data:
        print(f"\n[{tone}] 성공: {len(data)}건 / 실패: {failed}건")
        df = pd.DataFrame(data)
        output_filename = f"{output_path}/{tone.lower()}_chatml_dataset.csv"
        try:
            df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"데이터셋 생성 완료: {output_filename} ({len(df)} rows)")
        except Exception as e:
            print(f"저장 중 오류 발생: {e}")
    else:
        print(f"\n[{tone}] 데이터가 생성되지 않았습니다.")


def main():
    # parser = argparse.ArgumentParser(description="Qwen2.5용 ChatML 데이터셋 빌더")
    # parser.add_argument('--prompt', type=str, required=True, help='System prompt 파일 경로')
    # parser.add_argument('--instruction', type=str, required=True, help='User instruction 파일 경로')
    # parser.add_argument('--output', type=str, required=True, help='저장 폴더')
    # parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI 모델 이름 (기본: gpt-4o-mini)')


    parser = argparse.ArgumentParser(description="Qwen2.5용 ChatML 데이터셋 빌더")
    parser.add_argument('--prompt', type=str, default='/home/khw/Workspace/LLM/prompt/simple.txt', help='System prompt 파일 경로')
    parser.add_argument('--instruction', type=str, default='/home/khw/Workspace/LLM/data_aug/ANGRY.txt', help='User instruction 파일 경로')
    parser.add_argument('--output', type=str, default='/home/khw/Workspace/LLM/data', help='저장 폴더')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI 모델 이름 (기본: gpt-4o-mini)')

    args = parser.parse_args()
    build_chatml_dataset(args.prompt, args.instruction, args.output, args.model)


if __name__ == '__main__':
    main()
