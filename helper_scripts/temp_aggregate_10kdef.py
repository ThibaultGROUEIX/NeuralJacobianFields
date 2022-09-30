import os
import argparse
import sys
import json
import warnings

import numpy as np
import igl
def main(s_dir = '.',t_dir = '<target_folder>'):
    counter =0
    pairs = []
    groups = []
    for w in os.walk(s_dir):
        d,sd,f = w

        assert os.path.isdir(d)
        dirnum = os.path.basename(os.path.normpath(d))
        if not dirnum.isdigit():
            continue
        print(f'{counter} processed')
        print(f"in {dirnum}:")
        group =[]
        for i,fname in enumerate(f):
            #print(f'       {i}: fname')
            fname = os.path.join(d,fname)
            try:
                content = np.load(fname)
            except Exception as e:
                print(f'had an exception while loading numpy {e}, skipping')
                continue
            T = content['faces']
            if i == 0:
                oV = content['original_vertices']
                objfile = os.path.join(t_dir,f'{counter:08d}.obj')
                igl.write_obj(objfile, oV, T)
                try:
                    igl.read_obj(objfile)
                except Exception as e:
                    warnings.warn(f'{fname} source was problematic ({e}), skipping this whole example')
                    break
                source_counter = counter
                group.append(counter)
                counter += 1
            V = content['deformed_vertices']
            hs = content['original_handles']
            ht = content['deformed_handles']
            tname = os.path.join(t_dir, f'{counter:08d}')
            objfile = tname + '.obj'
            igl.write_obj(objfile, V, T)
            try:
                igl.read_obj(objfile)
            except Exception as e:
                warnings.warn(f'{fname} was problematic ({e}), skipping this whole example')
                continue
            np.save(f'{tname}_source_handles.npy',hs)
            np.save(f'{tname}_target_handles.npy',ht)
            counter += 1
            pairs.append((source_counter,counter))
            group.append(counter)
            counter += 1
        if len(group)>0:
            groups.append(group)
    print("writing jason")
    data = {'pairs': pairs, 'groups': group}
    with open(os.path.join(t_dir,'data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("read_dir", help="source dir to run over")
    parser.add_argument("write_dir", help="target dir to run over")
    parser.add_argument("--really_do_it", action="store_true")
    args = parser.parse_args()
    main(args.read_dir,args.write_dir)