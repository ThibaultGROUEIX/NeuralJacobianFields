import numpy as np
import fire
import json
from pathlib import Path 
import os

def main(path="/sensei-fs/users/groueix/db_train"):
    path = Path(path)
    num_files = len([entry for entry in os.listdir(path.as_posix()) if (os.path.join(path.as_posix(), entry).endswith('.obj'))])
    print(num_files)
    pairs = [(f"{i:08d}",f"{(i+1):08d}") for i in range(num_files) if i % 2 == 0]
    data = {"pairs":pairs}
    with open(path / "data.json", 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    fire.Fire(main)