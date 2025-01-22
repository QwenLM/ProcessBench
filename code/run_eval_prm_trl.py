# torchrun --nproc_per_node=4 run_eval_prm_trl.py 
import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


def collate_fn(batch, tokenizer, separator = '\n'):
    input_ids = []
    score_ids = []
    labels = []
    for i in batch:
        text = i['problem'] + separator
        input_idx = tokenizer(text, return_tensors='pt')['input_ids']
        score_ids.append([])
        for j in i['steps']:
            completion = j + '\n'
            completion_idx = tokenizer(completion, return_tensors='pt')['input_ids']
            input_idx = torch.cat([input_idx, completion_idx], dim=-1)
            score_ids[-1].append(input_idx.size(-1) - 1)
        labels.append(i['label'])
        input_ids.append(input_idx)
    
    # right pad input_ids
    pad_token_id = tokenizer.pad_token_id
    max_len = max([i.size(-1) for i in input_ids])
    for i, input_idx in enumerate(input_ids):
        input_ids[i] = torch.cat([
            input_idx.squeeze(), 
            torch.LongTensor(
                [pad_token_id] * (max_len - input_idx.size(-1))
            )
        ])
    input_ids = torch.stack(input_ids)

    return dict(
        input_ids=input_ids,
        labels=labels,
        score_ids=score_ids
    )
    
def find_first_zero(tensor):
    zeros = (tensor == 0).nonzero()
    return zeros[0].item() if zeros.numel() > 0 else -1

def gather_objects(data, accelerator):
    world_size = accelerator.num_processes
    if world_size == 1:
        return data
        
    all_data = [None] * world_size
    torch.distributed.all_gather_object(all_data, data)
    
    if accelerator.is_main_process:
        result = []
        for process_data in all_data:
            result.extend(process_data)
        return result
    return None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    bs = 24
    num_of_workers = 4
    separator = "\n"  # It's important to use the same separator as the one used during TRL training

    model_path = "/local_path_to_a_PRM_trained_by_TRL/qwen25-math-7b-instruct-PRM800k"
    model_name = model_path.split('/')[-1]

    configs = {
        'gsm8k': [207, 193], # error / correct num
        'math': [594, 406], 
        'olympiadbench': [661, 339], 
        'omnimath': [759, 241],
    }
    all_f1_scores = []
    save_dir = f'outputs/{model_name}'
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator()
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = accelerator.prepare(model)
    model.eval()

    for config, num in configs.items():
        dataset = load_dataset("Qwen/ProcessBench", split=config)
        sampler = None
        if accelerator.distributed_type == "MULTI_GPU":
            sampler = DistributedSampler(
                dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
            )
        dataloader = DataLoader(
            dataset, 
            batch_size=bs, 
            collate_fn=lambda x: x, 
            num_workers=num_of_workers,
            sampler=sampler,
            drop_last=False,
        )

        res_data = []
        for batch_ in tqdm(dataloader, disable=not accelerator.is_main_process):
            new_batch = deepcopy(batch_)

            batch = collate_fn(batch_, tokenizer, separator)
            input_ids = batch['input_ids'].to(accelerator.device)
            labels = batch['labels']
            score_ids = batch['score_ids']

            with accelerator.autocast(), torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            for i, score_id in enumerate(score_ids):
                label = labels[i]
                pred = torch.argmax(logits[i, score_id], dim=-1)
                prediction_step = find_first_zero(pred)
                new_batch[i]['prediction'] = prediction_step
                new_batch[i]['match'] = prediction_step == label
            
            res_data.extend(new_batch)
        
        accelerator.wait_for_everyone()
        gathered_data = gather_objects(res_data, accelerator)

        if accelerator.is_main_process:
            data1 = [e for e in gathered_data if e['label'] != -1]
            data2 = [e for e in gathered_data if e['label'] == -1]
            # dataset length check
            if len(data1) != num[0]:
                print(f'{config} error num mismatch: {len(data1)} != {num[0]}')
            if len(data2) != num[1]:
                print(f'{config} correct num mismatch: {len(data2)} != {num[1]}')
            
            with open(f'{save_dir}/{config}_error.jsonl', 'w') as f:
                for e in data1:
                    f.write(json.dumps(e) + '\n')
            with open(f'{save_dir}/{config}_correct.jsonl', 'w') as f:
                for e in data2:
                    f.write(json.dumps(e) + '\n')
            
            acc1 = np.mean([e['match'] for e in data1]) * 100
            acc2 = np.mean([e['match'] for e in data2]) * 100
            f1 = 2 * acc1 * acc2 / (acc1 + acc2)
            print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

            all_f1_scores.append(f1)

    if accelerator.is_main_process:
        print(f'ProcessBench. Average F1: {np.mean(all_f1_scores):.1f}')

    if accelerator.distributed_type == "MULTI_GPU":
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()