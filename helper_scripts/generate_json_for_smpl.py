import random
import json

def gen_random_pairs(groups,permutations_per_group):
    ret = []
    for group in groups:
        for _ in range(permutations_per_group):
            g2 = group.copy()
            random.shuffle(g2)
            ret.extend(list(zip(group,g2)))
    return ret

def gen_random_sources(groups,nsources):
    ret = []
    for gi,group in enumerate(groups):
        for _ in range(nsources):
            random.shuffle(group)
            source = group[0]
            for target in group[1:]:
                ret.append((source,target))
    return ret
    
if __name__ == '__main__':
    with open('data.json') as file:
        data = json.load(file)
    assert 'groups' in data
    groups = data['groups']
    ret = gen_random_sources(groups.copy(),20)
    data = {'pairs': ret,'groups':ret}
    with open('100_rand_sources.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # print(groups)
