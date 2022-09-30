from copy import copy
from genericpath import exists
from os import mkdir
from pathlib import Path
import shutil

root_dirs = [
    Path("/mnt/session_space/neural_deformations/lightning_logs/uv/test/version_23"),
    Path("/mnt/session_space/neural_deformations/lightning_logs/uv/test/version_24")
]
output_dir = Path("/mnt/session_space/neural_deformations/figure_uv/")
output_dir.mkdir(exist_ok =True)

ids = [
     ['180002', '180003', '180004', '180005', '180010', '180011', '180014', '180015', '180016', '180017', '180020', '180021', '180024', '180025', '180026', '180027', '180030', '180031', '180032', '180033', '180042', '180043', '180050', '180051', '180108', '180109'],
     ['001077', '001105', '001125', '001183', '001211', '001225', '001263', '001347', '001441', '001565', '001571', '001575', '001757', '001955']
]
names = ["10kValid", "LionVaseModles"]

for i in range(2):
    path = output_dir / names[i]
    path.mkdir(exist_ok =True)
    for item in ids[i]:
        item_plus_one = f"{(int(item) - 1):06d}"
        source_path = root_dirs[i] / f"{item}_from_{item_plus_one}_slim_summary.txt"
        target_path = path / f"{item}_from_{item_plus_one}_slim_summary.txt"
        if int(item)%2 == 0:
            print(f"{item} is even so skipping")
        else:
            shutil.copy(source_path, target_path)
            print(f"Success {item}")
