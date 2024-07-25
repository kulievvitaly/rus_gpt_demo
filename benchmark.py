import traceback

import openai
import time, json, random
from queue import Empty
import multiprocessing

HOST = '192.168.0.7'
PORT = 9000
PROTOCOL = 'http'



BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"

client = openai.OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)

# MODEL = 'NousResearch/Meta-Llama-3-8B-Instruct'
# MODEL = 'casperhansen/llama-3-70b-instruct-awq'
# MODEL = 'voxmenthe/gemma-2-27b-it-mlx-fp16'
# MODEL = 'cortecs/Meta-Llama-3-70B-Instruct-GPTQ-8b'
MODEL = 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'



prompt_extract = """
These are short, famous texts in English from classic sources like the Bible or Shakespeare. Some texts have word definitions and explanations to help you. Some of these texts are written in an old style of English. Try to understand them, because the English that we speak today is based on what our great, great, great, great grandparents spoke before! Of course, not all these texts were originally written in English. The Bible, for example, is a translation. But they are all well known in English today, and many of them express beautiful thoughts.
What is most known english book? Answer only name of the book. Do not write anything else.
"""


prompt_generate = """
Write detail story about kolobok.
"""

PROMPT_TYPE = 'prompt_extract'
# PROMPT_TYPE = 'prompt_generate'

print('PROMPT_TYPE', PROMPT_TYPE)
print('MODEL', MODEL)

'''
A100
# todo
========================
4x3090ti pl 300
extract +8-15%
awq
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	343,14	7,35	343,14	7,35	0,41
10	439,84	9,43	43,98	0,94	3,18
50	441,84	9,47	8,84	0,19	15,19
awq_marlin
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	370,62	8,15	370,62	8,15	0,38
10	503,49	11,44	50,35	1,14	2,74
50	509,51	11,01	10,19	0,22	13,16

generate +13-23%
awq
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	1,36	38,54	1,36	38,54	13,28
10	7,02	199,70	0,70	19,97	25,54
50	13,06	371,53	0,26	7,43	68,91
awq_marlin
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	1,67	47,37	1,67	47,37	10,81
10	7,99	226,86	0,80	22,69	22,48
50	15,14	426,12	0,30	8,52	58,90

'''


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
                    max_tokens=5,
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
            (10, 30),
            (50, 8),
            
            # (1, 50),
            # (2, 40),
            # (3, 30),
            # (4, 30),
            # (5, 30),
            # (6, 30),
            # (7, 30),
            # (8, 30),
            # (9, 30),
            # (10, 30),
            # (15, 20),
            # (20, 15),
            # (25, 10),
            # (30, 10),
            # (40, 8),
            # (50, 8),
            # (60, 8),
            
        ]
    else:
        worker_sample_list = [
            (1, 10),
            (10, 6),
            (50, 6),
            
            # (1, 10),
            # (2, 10),
            # (3, 10),
            # (4, 8),
            # (5, 8),
            # (6, 8),
            # (7, 8),
            # (8, 8),
            # (9, 8),
            # (10, 6),
            # (15, 6),
            # (20, 6),
            # (25, 6),
            # (30, 6),
            # (40, 6),
            # (50, 6),
            # (60, 6),
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
                if all([not p.is_alive() for p in process_list]):
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