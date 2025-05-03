import openai
import time, os
# from dotenv import load_dotenv
# load_dotenv()


HOST = '192.168.0.8'
PORT = 9000
PROTOCOL = 'http'
BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"
client = openai.OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)


# MODEL = 'NousResearch/Meta-Llama-3-8B-Instruct'
# MODEL = 'casperhansen/llama-3-70b-instruct-awq'
# MODEL = 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'
# MODEL = 'Qwen/Qwen2.5-72B-Instruct-AWQ'
# MODEL = 'Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ'
MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'


# prompt = '''
# I want to build application. analog of chatgpt. use PySide6 lib for gui. Client application should be writen on python.
# write client. it should contain 2 screens:
# 1. login screen
# - text field for api_key
# - check button. click on button check api_key with python-openai library.
# - save button. if check button get correct answer. save button should save api_key to default application library, show alert with succes and open next screen.
#
# 2. application screen
# - chat history with dialog
# - text field for text user input
# - send button. send text with python-openai library to api with stream support. result should be writen to chat history by token.
#
# write all code in to single file.
#
# Use lib versions from requirements.txt:
# PySide6
# openai==0.28
#
# '''


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
    print('=' * 120)
    print(f'elapsed {time.time() - timer} seconds.')
    
    print(chat_response.choices[0].message.content)
    





