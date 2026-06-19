import requests
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False

def send_discord_message(message: str, bot_name: str = "Python Bot"):
    # 현재 실행 중인 파이썬 파일의 부모 디렉토리를 찾습니다.
    cur_dir = os.path.dirname(__file__)
    env_path = os.path.join(cur_dir, '..', '..', '.env')

    # 해당 경로의 파일을 명시적으로 로드합니다.
    load_dotenv(dotenv_path=env_path, override=True)

    # 환경 변수에서 웹훅 URL 가져오기
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    # 웹훅 URL이 설정되었는지 확인
    if not webhook_url:
        print("Error: DISCORD_WEBHOOK_URL environment variable is not set.")
        return False

    # 2. 보낼 데이터 설정
    data = {
        "content": message,
        "username": bot_name  # name of bot
    }

    # 3. send POST request
    response = requests.post(webhook_url, json=data)

    # 4. 결과 확인
    if response.status_code == 204:
        print("Message Send succeed!")
        return True
    else:
        print(f"FAILED Sending: {response.status_code}")
        print(f"Response Body: {response.text}") # This will tell you EXACTLY what Discord didn't like
        return False

if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello from RunPod"
    send_discord_message(msg, bot_name="RunPod Bot")
