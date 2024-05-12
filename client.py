import openai
import time, os
from dotenv import load_dotenv

load_dotenv()


client = openai.OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url='https://api.rus-gpt.com/v1',
)

MODEL = 'NousResearch/Meta-Llama-3-8B-Instruct'

prompt = """
Calculate the expression: "2+5". answer in json format: {"answer": your_int_answer_here}. do not write anything else.
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
    





