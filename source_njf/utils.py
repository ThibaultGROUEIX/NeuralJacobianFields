import os
import shutil
import numpy as np
import torch

def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def fix_orientation(vertices, faces):
    from igl import bfs_orient
    new_faces, c = bfs_orient(faces)
    new_faces = new_faces.astype(int)

    # Edge case: only one face
    if len(new_faces.shape) == 1:
        new_faces = new_faces.reshape(1,3)

    volume = signed_volume(vertices, new_faces)
    if volume < 0:
        new_faces = np.fliplr(new_faces)
    return new_faces

def signed_volume(v, f):
    # Add up signed volume of tetrahedra for each face
    # If triangles, then one of these vertices is the origin
    if f.shape[1] == 3:
        f = np.hstack([f, np.ones(len(f)).reshape(len(f), 1) * len(v)]).astype(int)
        v = np.vstack([v, np.zeros(3).reshape(1, 3)])
    fverts = v[f]
    fvectors = fverts - fverts[:,3, None,:]
    # Triple scalar product
    volume = 1/6 * np.sum(np.sum(fvectors[:,0,:] * np.cross(fvectors[:,1,:], fvectors[:,2,:], axis=1), axis=1))
    return volume

# Pytorch: from UVs back out Jacobian matrices per face using local coordinates per triangle
# NOTE: uvtri input must be PER triangle
def get_jacobian(uvtri, local_tris):
    x = local_tris[:,:,0]
    y = local_tris[:,:,1]

    # NOTE: Below only valid when local tris maps the first vertex to (0,0)!!!
    d = (x[:, 1] * y[:, 2]).reshape(len(x), 1)

    # Construct J
    Jx = torch.column_stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]])
    Jy = torch.column_stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]])
    J = torch.matmul(torch.stack([Jx, Jy], dim=1), uvtri) * 1/d # F x 2 x 2

    return J

# Get Tutte embedding from IGL
def tutte_embedding(vertices, faces, fixclosed=False):
    import igl
    bnd = igl.boundary_loop(faces)

    ## If mesh is closed then we cut a seam if set
    if fixclosed and (bnd is None or len(bnd) == 0):
        from meshing.mesh import Mesh
        from meshing.edit import EdgeCut

        # TODO: cut out a triangle
        mesh = Mesh(vertices, faces)
        # for he in mesh.topology.faces[0].adjacentHalfedges():
        #     EdgeCut(mesh, he.index).apply()

        he2 = mesh.topology.halfedges[0].next.index
        EdgeCut(mesh, 0).apply()
        EdgeCut(mesh, he2).apply()
        bnd = igl.boundary_loop(faces)
    elif bnd is None:
        raise ValueError(f"tutte_embedding: mesh has no boundary and fixclosed is not set!")

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(vertices, bnd)

    ## Harmonic parametrization for the internal vertices
    assert not np.isnan(bnd).any(), f"NaN found in boundary loop!"
    assert not np.isnan(bnd_uv).any(), f"NaN found in tutte initialized UVs!"
    uv_init = igl.harmonic_weights(vertices, faces, bnd, np.array(bnd_uv, dtype=vertices.dtype), 1)

    return uv_init

# Convert each triangle into local coordinates: A -> (0,0), B -> (x2, 0), C -> (x3, y3)
def get_local_tris(vertices, faces, device=torch.device("cpu")):
    fverts = vertices[faces].to(device)
    e1 = fverts[:, 1, :] - fverts[:, 0, :]
    e2 = fverts[:, 2, :] - fverts[:, 0, :]
    s = torch.linalg.norm(e1, dim=1)
    t = torch.linalg.norm(e2, dim=1)
    angle = torch.acos(torch.sum(e1 / s[:, None] * e2 / t[:, None], dim=1))
    x = torch.column_stack([torch.zeros(len(angle)).to(device), s, t * torch.cos(angle)])
    y = torch.column_stack([torch.zeros(len(angle)).to(device), torch.zeros(len(angle)).to(device), t * torch.sin(angle)])
    local_tris = torch.stack((x, y), dim=-1).reshape(len(angle), 3, 2)
    return local_tris