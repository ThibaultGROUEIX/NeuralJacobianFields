import json
import random
import generate_json_for_smpl
def source_target_sequence(fname,n):
    pairs = [(counter,counter+1) for counter in range(0,n,2)]
    save_it(fname,pairs)
def staggered_sources(fname,n,m,source_to_source = False):
    pairs = []
    for i in range(n):
        if i % m == 0:
            source = i
            if not source_to_source:
                continue
        pairs.append((source,i))
    save_it(fname,pairs)
def staggered_groups(fname,n,m,source_per_group):
    pairs = []
    for i in range(0,n,m):
        group = [j for j in range(i,i+m)]
        for i in range(source_per_group):
            g2 = group.copy()
            random.shuffle(g2)
            pairs.extend([(g2[0],g2[i]) for i in range(1,len(group))])
    save_it(fname,pairs)
def random_pairs(fname,n,loops):
    r = [i for i in range(n)]
    p =  generate_json_for_smpl.gen_random_pairs([r],loops)
    save_it(fname, p)
def all_to_one(fname,n,s=0):
    pairs = [(i,s) for i in range(1,n)]
    save_it(fname,pairs)
def save_it(fname,pairs):
    if fname == 'print':
        print(pairs)
        return
    with open(fname,'w') as f:
        json.dump({'pairs':pairs},f)


if __name__ == '__main__':
    all_to_one('print',127,0)

