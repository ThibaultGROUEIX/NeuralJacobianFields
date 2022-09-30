import numpy as np
import igl
V,T,_ = igl.read_off('data/faust/faust_000.off')
import trimesh
viz = trimesh.visual.TextureVisuals(uv=V[:,:2])
m = trimesh.Trimesh(vertices=V,faces=T,visual=viz)
s = trimesh.exchange.obj.export_obj(m)
with open('testi.obj','w') as f:
    f.write(s)