import json
import random
from copy import deepcopy
random.seed(0)

source_paths = (
    'cleanned_cmrc2018_dev.json',
    'cleanned_cmrc2018_train.json'
)

for source_path in source_paths:
    with open(source_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    source_data = [json.loads(line.strip()) for line in lines]
    target_data = []
    
    for pair in source_data:
        tmp = deepcopy(pair)
        tmp.pop('st')
        tmp.pop('ed')
        tmp.pop('answer')
        posi, nega = deepcopy(tmp), deepcopy(tmp)
        posi.update({"answerable": 1})
        nega.update({"answerable": 0})
        nega.update({"question": random.choice(source_data)['question']})

        target_data.append(posi)
        target_data.append(nega)
    
    target_path = source_path.replace('cleanned', 'cleanned_binary_from')
    with open(target_path, 'w', encoding='utf-8') as f:
        for trip in target_data:
            f.write(json.dumps(trip, ensure_ascii=False)+'\n')