import traceback

import openai
import time, json, random
from queue import Empty
import multiprocessing

HOST = '192.168.0.7'
PORT = 9000
PROTOCOL = 'http'

# HOST = 'api.rus-gpt.com'
# PORT = 443
# PROTOCOL = 'https'

# HOST = '81.94.150.40'
# PORT = 8000
# PROTOCOL = 'http'

# HOST = 'pmg-matcher-gpu-svc-01.el.wb.ru'
# PORT = 8080
# PROTOCOL = 'http'

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
5 gpu. pl 300.
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	1,27	36,08	1,27	36,08	14,19
10	6,98	198,68	0,70	19,87	25,77
50	11,68	332,22	0,23	6,64	68,60
global_elapsed 708.1096098423004 seconds

4 gpu connected. correct ordering. pl 300.
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	1,35	38,45	1,35	38,45	13,31
10	7,08	201,35	0,71	20,14	25,43
50	11,09	315,32	0,22	6,31	68,02
global_elapsed 693.3846619129181 seconds

4 gpu connected. correct ordering. pl 300. enforce-eager
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	0,74	20,91	0,74	20,91	24,48
10	6,56	186,67	0,66	18,67	25,60
50	11,70	332,68	0,23	6,65	66,54
global_elapsed 797.1303713321686 seconds

'''


'''
A100
extract +57-89%
awq
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	246,70	5,29	246,70	5,29	0,57
10	539,56	11,56	53,96	1,16	2,55
50	659,82	14,14	13,20	0,28	9,92
awq_marlin
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	465,70	9,98	465,70	9,98	0,30
10	1006,55	21,57	100,65	2,16	1,23
50	1039,58	22,43	20,79	0,45	5,43

generate +41-130%
awq
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	0,65	18,46	0,65	18,46	27,74
10	4,45	126,72	0,45	12,67	35,04
50	7,13	202,85	0,14	4,06	107,42
awq_marlin
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	0,92	26,11	0,92	26,11	19,61
10	7,40	210,40	0,74	21,04	22,74
50	16,41	466,85	0,33	9,34	45,90

========================
4x3090ti pl 300
extract +0-9%
awq
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	335,58	7,19	335,58	7,19	0,42
10	439,56	9,42	43,96	0,94	3,14
50	426,44	9,14	8,53	0,18	15,36
awq_marlin
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	360,07	7,72	360,07	7,72	0,39
10	441,80	9,47	44,18	0,95	1,79
50	467,99	10,03	9,36	0,20	12,00

awq_marlin_eager


generate +17-31%
awq
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	1,34	38,04	1,34	38,04	13,46
10	6,81	193,58	0,68	19,36	26,45
50	11,03	313,70	0,22	6,27	70,06
awq_marlin
workers	prompt, tps	completion, tps	prompt_worker, tps	completion_worker, tps	latency, sec
1	1,66	47,25	1,66	47,25	10,83
10	7,98	227,11	0,80	22,71	22,54
50	14,82	412,37	0,30	8,25	59,09

awq_marlin_eager

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
                # print([not p.is_alive() for p in process_list])
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