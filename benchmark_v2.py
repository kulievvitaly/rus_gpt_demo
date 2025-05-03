import traceback
import openai
import time, json, random
from queue import Empty
import multiprocessing

# Server configuration
HOST = '192.168.0.10'
PORT = 8000
PROTOCOL = 'http'
BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"

client = openai.OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)

# Model configuration
# MODEL = 'Qwen/Qwen3-1.7B'
MODEL = 'Qwen/Qwen3-8B'

# Test cases with carefully constructed prompts
TEST_CASES = {
    # 'case1': {
    #     'description': '256 input tokens, 1 output token',
    #     'prompt': """This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. Answer with the letter A.""",
    #     'max_tokens': 1
    # },
    'case2': {
        'description': '1 input token, 256 output tokens',
        'prompt': """Write:""",
        'max_tokens': 256
    },
    # 'case3': {
    #     'description': '256 input tokens, 256 output tokens',
    #     'prompt': """This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. This is a test prompt designed to be approximately 256 tokens in length. It contains repetitive text to reach that token count. This sentence is repeated multiple times to achieve the desired length. Write a detailed story about space exploration.""",
    #     'max_tokens': 256
    # },
}


def worker(arg):
    queue, samples, test_case = arg
    prompt = TEST_CASES[test_case]['prompt']
    max_tokens = TEST_CASES[test_case]['max_tokens']

    for i in range(samples):
        try:
            time.sleep(random.random() * 0.001)
            timer = time.time()

            chat_response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=max_tokens,
            )

            queue.put({
                'prompt_tokens': chat_response.usage.prompt_tokens,
                'completion_tokens': chat_response.usage.completion_tokens,
                'elapsed': (time.time() - timer)
            })
        except:
            print(traceback.format_exc())


def run_benchmark(test_case, worker_sample_list):
    print(f"\n--- Running benchmark for {TEST_CASES[test_case]['description']} ---")
    print('\t'.join([
        'workers',
        'prompt, tps',
        'completion, tps',
        'total, tps',
        'prompt_worker, tps',
        'completion_worker, tps',
        'total_worker, tps',
        'latency, sec'
    ]))

    for workers, samples in worker_sample_list:
        process_list = []

        queue = multiprocessing.Queue()
        prompt_tokens_list = []
        completion_tokens_list = []
        elapsed_list = []

        for i in range(workers):
            p = multiprocessing.Process(target=worker, args=((queue, samples, test_case),))
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

        prompt_tps = sum(prompt_tokens_list) / elapsed
        completion_tps = sum(completion_tokens_list) / elapsed
        total_tps = prompt_tps + completion_tps
        prompt_worker_tps = prompt_tps / workers
        completion_worker_tps = completion_tps / workers
        total_worker_tps = total_tps / workers
        avg_latency = sum(elapsed_list) / len(elapsed_list)

        print(('\t'.join([
            str(workers),
            '%.2f' % prompt_tps,
            '%.2f' % completion_tps,
            '%.2f' % total_tps,
            '%.2f' % prompt_worker_tps,
            '%.2f' % completion_worker_tps,
            '%.2f' % total_worker_tps,
            '%.2f' % avg_latency,
        ])).replace('.', ','))  # replace dot for google docs


if __name__ == '__main__':
    print(f"Running VLLM Inference Benchmark")
    print(f"MODEL: {MODEL}")
    print(f"BASE_URL: {BASE_URL}")

    # Define the worker samples for benchmarking
    worker_sample_list = [
        (1, 10),
        (2, 10),
        (3, 10),
        (4, 9),
        (5, 9),
        (6, 8),
        (7, 8),
        (8, 8),
        (9, 8),
        (10, 8),
        (15, 7),
        (20, 7),
        (25, 6),
        (30, 6),
        (40, 6),
        (50, 6),
        (100, 6),
        (200, 6),
    ]

    # Run benchmarks for all three test cases
    global_timer = time.time()

    for case in TEST_CASES.keys():
        run_benchmark(case, worker_sample_list)

    global_elapsed = time.time() - global_timer
    print(f'\nTotal benchmark time: {global_elapsed:.2f} seconds')
