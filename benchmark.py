import traceback

import openai
import time, json, random
from queue import Empty
import multiprocessing

HOST = '192.168.0.7'
PORT = 9001
PROTOCOL = 'http'

# HOST = 'api.rus-gpt.com'
# PORT = 443
# PROTOCOL = 'https'

BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"

client = openai.OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)

MODEL = 'NousResearch/Meta-Llama-3-8B-Instruct'
# MODEL = 'casperhansen/llama-3-70b-instruct-awq'

prompt_extract = """
These are short, famous texts in English from classic sources like the Bible or Shakespeare. Some texts have word definitions and explanations to help you. Some of these texts are written in an old style of English. Try to understand them, because the English that we speak today is based on what our great, great, great, great grandparents spoke before! Of course, not all these texts were originally written in English. The Bible, for example, is a translation. But they are all well known in English today, and many of them express beautiful thoughts.
What is most known english book? Answer only name of the book. Do not write anything else.
"""


prompt_generate = """
Write detail story about kolobok.
"""

# PROMPT_TYPE = 'prompt_extract'
PROMPT_TYPE = 'prompt_generate'

print('PROMPT_TYPE', PROMPT_TYPE)
print('MODEL', MODEL)


def worker(arg):
    queue, samples = arg
    for i in range(samples):
        try:
            time.sleep(random.random() * 0.001)
            timer = time.time()
            if PROMPT_TYPE == 'prompt_extract':
                chat_response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": prompt_extract},
                    ],
                    temperature=0,
                    max_tokens=512,
                )
            elif PROMPT_TYPE == 'prompt_generate':
                chat_response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": prompt_generate}
                    ],
                    temperature=0,
                    max_tokens=512,
                )
            else:
                raise Exception('promt not found')
                
            # s = chat_response.choices[0].message.content
            # print(s)
            
            # print(f'elapsed {time.time() - timer} seconds. '
            #       f'prompt_tokens: {chat_response.usage.prompt_tokens} '
            #       f'completion_tokens: {chat_response.usage.completion_tokens} ')

            queue.put({
                'prompt_tokens': chat_response.usage.prompt_tokens,
                'completion_tokens': chat_response.usage.completion_tokens,
                'elapsed': (time.time() - timer)
            })
        except:
            print(traceback.format_exc())


if __name__ == '__main__':
    print('\t'.join(['workers', 'prompt, tps', 'completion, tps', 'prompt_worker, tps', 'completion_worker, tps', 'latency, sec']))
    
    if PROMPT_TYPE == 'prompt_extract':
        worker_sample_list = [
            (1, 50),
            (2, 40),
            (3, 30),
            (4, 30),
            (5, 30),
            (6, 30),
            (7, 30),
            (8, 30),
            (9, 30),
            (10, 30),
            (15, 20),
            (20, 15),
            (25, 10),
            (30, 10),
            (40, 8),
            (50, 8),
            (60, 8),
        ]
    else:
        worker_sample_list = [
            (1, 10),
            (2, 10),
            (3, 10),
            (4, 8),
            (5, 8),
            (6, 8),
            (7, 8),
            (8, 8),
            (9, 8),
            (10, 6),
            (15, 6),
            (20, 6),
            (25, 6),
            (30, 6),
            (40, 6),
            (50, 6),
            (60, 6),
        ]
    global_timer = time.time()
    for workers, samples in worker_sample_list:
    
        process_list = []
        
        queue = multiprocessing.Queue()
        prompt_tokens_list = []
        completion_tokens_list = []
        elapsed_list = []
        
        for i in range(workers):
            p = multiprocessing.Process(target=worker, args=((queue, samples),))
            p.start()
            process_list.append(p)
    
        timer = time.time()
        while True:
            try:
                # print([not p.is_alive() for p in process_list])
                if any([not p.is_alive() for p in process_list]):
                    break
                queue_dict = queue.get(timeout=0.01)
    
                prompt_tokens_list.append(queue_dict['prompt_tokens'])
                completion_tokens_list.append(queue_dict['completion_tokens'])
                elapsed_list.append(queue_dict['elapsed'])
                
            except Empty:
                pass
            except:
                print(traceback.format_exc())
        
        elapsed = time.time() - timer
        
        print(('\t'.join([
            str(workers),
            '%.2f' % (sum(prompt_tokens_list) / elapsed),
            '%.2f' % (sum(completion_tokens_list) / elapsed),
            '%.2f' % (sum(prompt_tokens_list) / elapsed / workers),
            '%.2f' % (sum(completion_tokens_list) / elapsed / workers),
            '%.2f' % (sum(elapsed_list) / len(elapsed_list)),
        ])).replace('.', ','))  # replace dot for google docs
    
    global_elapsed = time.time() - global_timer
    print(f'global_elapsed {global_elapsed} seconds')