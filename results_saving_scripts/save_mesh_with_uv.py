import trimesh
def write_with_uv(fname,V,T,uv):
    viz = trimesh.visual.TextureVisuals(uv=uv)
    m = trimesh.Trimesh(vertices=V,faces=T,visual=viz)
    s = trimesh.exchange.obj.export_obj(m)
    with open(fname,'w') as f:
        f.write(s)