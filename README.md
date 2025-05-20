# ProcessBench

📄 [**[paper]**](https://huggingface.co/papers/2412.06559) 🤗 [**[data]**](https://huggingface.co/datasets/Qwen/ProcessBench)

This is the official repository for **ACL 2025** paper "[ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://huggingface.co/papers/2412.06559)"

If you find this work relevant or helpful to your work, please kindly cite us:

```latex
@inproceedings{
  processbench,
  title={ProcessBench: Identifying Process Errors in Mathematical Reasoning}, 
  author={
    Chujie Zheng and Zhenru Zhang and Beichen Zhang and Runji Lin and Keming Lu and
    Bowen Yu and Dayiheng Liu and Jingren Zhou and Junyang Lin
  },
  booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics
},
  year={2025}
}
```

## Data Usage

You can use the following code to preview the ProcessBench data:

```python
import json
from datasets import load_dataset

dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
print(json.dumps(dataset[0], indent=2))

# Expected output:
"""
{
  "id": "gsm8k-0",
  "generator": "Qwen2-7B-Instruct",
  "problem": "Sue lives in a fun neighborhood...",
  "steps": [
    "To find out how many more pink plastic flamingos were out than...",
    ...
  ],
  "final_answer_correct": false,
  "label": 1
}
"""
```

## Evaluation

You can refer to the [code](./code) folder for the evaluation code and the prompt templates we use in this work
