"""
Download the monkey business dataset and save it to a pickle file.

To load data:
import pickle
with open('monkey_business.pkl', 'rb') as f:
    data_dict = pickle.load(f)
data_dict['data'].shape = (num_problems, num_trials)
data_dict['question']: List[str] = problem texts
"""
import datasets
import numpy as np
from collections import defaultdict

data_dict = defaultdict(dict)
from time import perf_counter
for cfg in datasets.get_dataset_config_names("ScalingIntelligence/monkey_business"):
    print(cfg)
    start = perf_counter()
    ds = datasets.load_dataset("ScalingIntelligence/monkey_business", cfg, split='test')
    data = [ds[i]['is_corrects'] for i in range(len(ds))]
    data = np.array(data)
    print('data shape', data.shape)

    data_dict[cfg]['data'] = data
    data_dict[cfg]['question'] = ds['question']
    data_dict[cfg]['prompt'] = ds['prompt']
    end = perf_counter()
    print(f'time taken: {end - start} seconds')

# save data_dict to pickle
import pickle
with open('monkey_business.pkl', 'wb') as f:
    pickle.dump(data_dict, f)