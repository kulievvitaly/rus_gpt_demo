import openai
import time, os
# from dotenv import load_dotenv
# load_dotenv()


# HOST = 'pmg-matcher-gpu-svc-01.el.wb.ru'
# PORT = 8000
# PROTOCOL = 'http'

# HOST = '192.168.0.8'
# PORT = 9000
# PROTOCOL = 'http'

# HOST = '62.68.147.88'
# PORT = 9000
# PROTOCOL = 'http'

HOST = '192.168.0.10'
PORT = 8000
PROTOCOL = 'http'

BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"
client = openai.OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)


# MODEL = 'Qwen/Qwen2.5-VL-72B-Instruct-AWQ'
# MODEL = 'Qwen/Qwen2-VL-72B-Instruct-AWQ'
MODEL = 'Qwen/Qwen3-8B'

prompt = 'write:'


def get_image_from_file(file_path: str) -> str:
    import base64

    # Read image file in binary mode
    with open(file_path, 'rb') as image_file:
        # Read binary content
        binary_data = image_file.read()

    # Encode to base64 and convert to string
    encoded_string = base64.b64encode(binary_data).decode('utf-8')

    return encoded_string


if __name__ == '__main__':
    timer = time.time()
    image_path = "/home/me/Downloads/am5.jpg"
    image = get_image_from_file(image_path)
    chat_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                ]
            }
        ],
        temperature=0,
        max_tokens=256,
    )
    print(chat_response)
    print('=' * 120)
    print(f'elapsed {time.time() - timer} seconds.')
    
    print(chat_response.choices[0].message.content)
    





