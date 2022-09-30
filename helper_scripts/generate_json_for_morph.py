import random
import json
import numpy as np

max_val = 199998
num_targets = 32

def get_random(num):
    return 2*np.random.choice(max_val/2, size=num)

# pairs_train = [(f"{(2*i):08d}", f"{(2*(random.randint(0,max_val//2))):08d}") for i in range(max_val//2)]
# for i in range(num_targets-1):
#     pairs_train = pairs_train + [(f"{(2*i):08d}", f"{(2*(random.randint(0,max_val//2))):08d}") for i in range(max_val//2)]

# data = {'pairs': pairs_train}
# with open('data_train.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

pairs_test = [(f"{(2*i):08d}", f"{(2*(random.randint(0,1000//2))):08d}") for i in range(1000//2)]

data = {'pairs': pairs_test}
with open('data_test.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)