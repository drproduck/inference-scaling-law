from collections import defaultdict
from time import perf_counter
import datasets
import numpy as np
models = ['r1-qwen7b', 'r1-qwen14b', 'qwen3-8b', 'qwen3-14b']
dsets = ['aime25', 'hmmtfeb25', 'gpqa-diamond', 'livecodebench-subset-v6']

def pass_at_k(list_scores):
    return np.mean([np.any(scores) for scores in list_scores])

data_dict = defaultdict(dict)
for model in models:
    for ds in dsets:
        start = perf_counter()
        dataset = datasets.load_dataset(f'drproduck/{model}-{ds}-n128', split='train', columns=['prompt', 'scores'])
        print(f'{model} {ds} mean={np.mean(dataset["scores"])}, pass_at_128={pass_at_k(dataset["scores"])}')

        data = np.array([dataset[i]['scores'] for i in range(len(dataset))])
        print(data.shape)

        cfg = f'{ds}_{model}'

        data_dict[cfg]['data'] = data
        data_dict[cfg]['prompt'] = dataset['prompt']
        end = perf_counter()
        print(f'time taken: {end - start} seconds')

# save data_dict to pickle
import pickle
with open('r1_reasoning.pkl', 'wb') as f:
    pickle.dump(data_dict, f)