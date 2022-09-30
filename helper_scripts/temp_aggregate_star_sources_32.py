import os
import argparse
import sys
import shutil
files_to_copy = ['gt.obj','handles.npy']
# txt = "hi 8888 "
# nums = [int(s) for s in txt.split() if s.isdigit()]
# assert len(nums) == 1
# num = nums[0]
# output_name = f'{num:08d}'
# print(output_name)
def main(s_dir = '.',t_dir = '<target_folder>',doit = False):
    counter =0
    for w in os.walk(s_dir):

        full_src = w[0]
        without_extra_slash = os.path.normpath(full_src)
        rel_src = os.path.basename(without_extra_slash)
        if not rel_src.startswith('star_'):
            continue
        numstr = rel_src.replace('star_','')
        if not numstr.isdigit():
            continue
        index = int(numstr)
        if index % 32 != 0:
            continue
        out_name = f't_pose_of_{index:08d}'
        copy_and_rename('source.obj', full_src, out_name, t_dir, True, doit)
        if counter % 100 == 0:
            print(f"========== {counter} files  so far =======")
        counter += 1


def copy_and_rename(org_fname, src_id, trgt_prefix, t_dir,the_mesh,doit):
    src_file = os.path.join(src_id, org_fname)
    if  the_mesh:
        tname = trgt_prefix + '.obj'
    else:
        tname = f'{trgt_prefix}_{org_fname}'
    target_file = os.path.join(t_dir, tname)
    if doit:
        print(f"copying {src_file} to {target_file}")
        shutil.copy(src_file,target_file)
    else:
        print(f"I would copy {src_file} to {target_file} if this weren't a dry run")
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("read_dir", help="source dir to run over")
    parser.add_argument("write_dir", help="target dir to run over")
    parser.add_argument("--really_do_it", action="store_true")
    # parser.add_argument('dir', nargs='?', default='data/faust')
    args = parser.parse_args()
    main(args.read_dir,args.write_dir,args.really_do_it)