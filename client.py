import openai
import time, os
from dotenv import load_dotenv

load_dotenv()


HOST = 'pmg-matcher-gpu-svc-01.el.wb.ru'
PORT = 8080
PROTOCOL = 'http'

BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"

client = openai.OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=BASE_URL,
)

# MODEL = 'NousResearch/Meta-Llama-3-8B-Instruct'
# MODEL = 'casperhansen/llama-3-70b-instruct-awq'
# MODEL = 'llama-3-70b'
MODEL = 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'


prompt = """
Calculate the expression: "16+16". answer in json format: {"answer": your_int_answer_here}. do not write anything else.
"""


if __name__ == '__main__':
    timer = time.time()
    chat_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    print(chat_response)
    print(f'elapsed {time.time() - timer} seconds.')
    
    print(chat_response.choices[0].message.content)
    





