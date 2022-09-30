#0.6057636141775758
import numpy as np
import igl
# f = 'C:/Users/aigerman/Dropbox/ndef/star/starstuff/S102415/102415.obj'
# _,_,_,T,_,_ = igl.read_obj(f)
# for i in range(100):
#     sdir = f'C:/for_reverse_map_processed/star_manipulate/{i:08d}/'
#     V = np.load(sdir+'vertices.npy')
#     #T = np.load(sdir+'triangles.npy')
#     tdir = f'C:/Users/aigerman/Dropbox/ndef/star/bigbuck_bunny/star_poses/{i:08d}.obj'
#     igl.write_obj(tdir, V,T)

for f in ['beast.obj','big-buck-bunny.obj']:
    V,_,_,F,_,_ = igl.read_obj('C:/quickone/'+f)
    V = V - np.mean(V,0)
    n = np.mean(np.sum(np.sqrt(V**2),1))
    V /= n
    if f == 'big-buck-bunny.obj':
        V *= 0.6

        V[:,1]+=0.2
    else:
        V *= 0.8
    igl.write_obj(f'C:/final_for_star/{f}',V,F)
# for f in ['C:/for_reverse_map/0_Armadillo.obj']:
#     V,_,_,F,_,_ = igl.read_obj(f)
#     V = V - np.mean(V,0)
#     n = np.mean(np.sum(np.sqrt(V**2),1))
#     V /= n
#     V *= 0.8
#     V[:,0] = - V[:,0]
#     V[:,2]= - V[:,2]
#     igl.write_obj('C:/for_reverse_map/1_Armadillo.obj',V,F)


#
# sdir = f'C:/for_reverse_map_processed//00000010/'
# V = np.load(sdir+'vertices.npy')
# T = np.load(sdir+'faces.npy')
# tdir = f'C:/for_reverse_map/star.obj'
# igl.write_obj(tdir, V,T)