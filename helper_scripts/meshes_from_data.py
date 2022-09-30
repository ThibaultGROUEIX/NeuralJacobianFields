import numpy as np
import os
import igl
import sys
def do_it(tdir,fname):
    v = np.load(os.path.join(tdir,'vertices.npy'))
    f = np.load(os.path.join(tdir,'faces.npy'))
    igl.write_obj(fname,v,f)
if __name__ == '__main__':
    do_it(sys.argv[1],sys.argv[2])