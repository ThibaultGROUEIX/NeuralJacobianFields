import os
import sys
import trimesh
import numpy as np
from pathlib import Path
from os.path import join

def obj_to_npy(path):
    vertices = np.load(join(path , "vertices.npy")).squeeze()
    faces = np.load(join(path ,'..', "faces.npy")).squeeze()
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.export(f"{path}.obj")

if __name__ == "__main__":
    obj_to_npy(sys.argv[1])