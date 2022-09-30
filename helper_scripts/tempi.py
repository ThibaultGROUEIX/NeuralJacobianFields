import os
f = 'C:/Users/aigerman/Dropbox/ndef/uv/uv_blender'
fs = os.listdir(f)
b = {}
for f in fs:
    b[f[:6]] =1
print(list(b.keys()))
# import shutil
# files = [180002,
#          180004,
#             180010,
#         180014,
#         180016,
#         180020,
#         180026,
#         180024,
#         180030,
#         180032,
#         180042,
#         180050,
#         180108]
# for file in files:
#     shutil.copytree(file, '/mnt/session_space/uv_blender/')
import os
import igl
# dir = 'data/for_uv'
# for f in os.listdir(dir):
#     V,_,_,F,_,_ = igl.read_obj('data/for_uv/'+f)
#     V,F,_,_ = igl.remove_unreferenced(V,F)
#     igl.write_obj('data/for_uv/'+f,V,F)