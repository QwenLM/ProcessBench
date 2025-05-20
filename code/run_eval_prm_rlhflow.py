# example code to evaluate the `RLHFlow/Llama3.1-8B-PRM-Mistral-Data` PRM
"""
Suppose you have launch an vllm server, e.g., through:

```
vllm serve \
	RLHFlow/Llama3.1-8B-PRM-Mistral-Data \
	--served-model-name Llama3.1-8B-PRM-Mistral-Data \
    --port 8000 \
    --tensor-parallel-size 8 \
    --dtype auto \
    --api-key token-abc123 \
    --enable-prefix-caching
```

Then you can run the following script to evaluate the model.
"""

import json
import os
from multiprocessing import Pool

import numpy as np
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    os.makedirs('outputs/Llama3.1-8B-PRM-Mistral-Data', exist_ok=True)

    def single_process(d):
        steps = d['steps']
        messages = []
        for sdx, step in enumerate(steps):
            if sdx == 0:
                messages.append({'role': 'user', 'content': d['problem'] + '\n\n' + step})
            else:
                messages.append({'role': 'user', 'content': step})
            completion = client.chat.completions.create(
                model='Llama3.1-8B-PRM-Mistral-Data',
                messages=messages,
                n=1,
                temperature=0.,
                max_tokens=1,
            )
            judgment = completion.choices[0].message.content.strip().lower().startswith('+')
            if not judgment:
                return sdx
            messages.append({'role': 'assistant', 'content': '+'})
        return -1

    configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    for config in configs:
        input_data = load_dataset('Qwen/ProcessBench', split=config)
        with Pool(32) as p:
            predictions = list(tqdm(p.imap(single_process, input_data), total=len(input_data),
                                    desc=f'Processing {config}', dynamic_ncols=True))
        
        res_data = []
        for idx, d in enumerate(input_data):
            new_d = d.copy()
            new_d['prediction'] = predictions[idx]
            new_d['match'] = predictions[idx] == d['label']
            res_data.append(new_d)
        
        data1 = [e for e in res_data if e['label'] != -1]
        data2 = [e for e in res_data if e['label'] == -1]
        with open(f'outputs/Llama3.1-8B-PRM-Mistral-Data/{config}_error.jsonl', 'w') as f:
            for e in data1:
                f.write(json.dumps(e) + '\n')
        with open(f'outputs/Llama3.1-8B-PRM-Mistral-Data/{config}_correct.jsonl', 'w') as f:
            for e in data2:
                f.write(json.dumps(e) + '\n')
        
        acc1 = np.mean([e['match'] for e in data1]) * 100
        acc2 = np.mean([e['match'] for e in data2]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

if __name__ == '__main__':
    main()
