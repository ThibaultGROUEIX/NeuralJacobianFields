#uv
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
#     shutil.copytree(f'S{file}/', f'/mnt/session_space/uv_blender/{file}')

#smpl
# import shutil
# files = [89386,
#          90032,
#         90020,
#          89976,
#          89953,
#          89940,
#          89340,
#          89175]
# for file in files:
#     shutil.copytree(f'S{file:06d}/', f'/mnt/session_space/smpl_test/{file}')

#star
import shutil
files = [
            103317,103115,
         103232]
for file in files:
    shutil.copytree(f'S{file:06d}/', f'/mnt/session_space/star_handles/{file}')
