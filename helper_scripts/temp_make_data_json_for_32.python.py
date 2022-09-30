import os
import json
import sys
def make(folder):
    fs = os.listdir(folder)
    assert len(fs)>0
    fs.sort()
    inds = {f:i for i,f in enumerate(fs)}
    pairs = []
    last = -1
    missing = []
    for i,dir in enumerate(fs):
        dir = os.path.basename(os.path.normpath(dir))
        if len(dir) != 8 or not dir.isdigit():
            continue
        ind = int(dir)
        if ind-1 != last:
              print("missing file, {ind} but last ind was {last}")
              missing.append(ind-1)
        last = ind
        source = f't_pose_of_{32*int(ind/32):08d}'
        #print(f'{dir}---> {source}')
        pairs.append((inds[source],ind))
    with open(os.path.join(folder,'data.json'),'w') as f:
        json.dump({'pairs':pairs},f)
    print(missing)
if __name__ == "__main__":
	make(sys.argv[1])
