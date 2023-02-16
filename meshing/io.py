import numpy as np
from .union_find import UnionFind


class PolygonSoup():
    """
    We define a triangular polygon soup as a collection of spatial points (vertices),
    and their connectivity information.
    The fields are:
        vertices: [N, 3]
        indices: [M, 3] where each row is the indices of the 3 vertices that make up a face

    This can be read from and written to in various file formats e.g obj, stl
    """
    def __init__(self, vertices, indices, uvs=None, face_uv=None, normals=None, face_normals=None):
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)
        if uvs is not None: 
            uvs = np.asarray(uvs, dtype=np.float32)
        if face_uv is not None: 
            face_uv = np.asarray(face_uv, dtype=np.uint32)
        if normals is not None: 
            normals = np.asarray(normals, dtype=np.float32)
        if face_normals is not None: 
            face_normals = np.asarray(face_normals, dtype=np.uint32)
            
        assert vertices.shape[1] == 3, "`vertices` must be an Nx3 array"
        assert indices.shape[1] == 3, "`faces` must be an Mx3 array"
        self.vertices = vertices
        self.indices = indices
        self.uvs = uvs 
        self.face_uv = face_uv
        self.normals = normals 
        self.face_normals = face_normals 

    def __eq__(self, other):
        # need to sort the fields so that comparison makes sense
        raise NotImplementedError()

    def nConnectedComponents(self):
        """should have the ability to separate meshes into components at soup stage"""
        uf = UnionFind(len(self.vertices))
        for triplet in self.indices:
            a, b, c = triplet
            uf.merge_into(a, b)
            uf.merge_into(a, c)
        return uf.n_components

    def split_into_soups(self):
        raise NotImplementedError()

    @classmethod
    def from_obj(cls, fname):
        """
        # An obj file looks like:
        v 0.123 0.234 0.345
        vn 0.707 0.000 0.707
        f 1 2 3
        # each line could be vertex_index/texture_index. Only need vertex_index
        f 3/1 4/2 5/3

        We can recompute normals "vn" ourselves
        """
        vertices = []
        indices = []
        f_uv = [] 
        f_normals = [] 
        normals = [] 
        uvs = [] 
        with open(fname, "r") as f_handle:
            for line in f_handle:
                line = line.strip()
                tokens = line.split(" ")
                identifier = tokens[0]

                if identifier == "v":
                    vertices.append(
                        [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                    )
                elif identifier == "f":
                    assert len(tokens) == 4,\
                        f"only triangle meshes are supported, got face index {line}"

                    face_indices = []
                    face_uv = [] 
                    face_normals = [] 
                    for i in range(3): 
                        inx = tokens[i+1].split("/")
                        assert len(inx) > 0, f"Expected face indices to have at least one value in line {line}"
                        if len(inx) > 0: 
                            face_indices.append(int(inx[0])-1)
                        if len(inx) > 1 and inx[1] != "":
                            face_uv.append(int(inx[1])-1)
                        if len(inx) > 2 and inx[2] != "": 
                            face_normals.append(int(inx[2])-1)
                        
                    # for i in range(3):
                    #     inx = tokens[1 + i].split("/")[0]  # throw away texture index, etc
                    #     inx = int(inx)
                    #     # NOTE obj index is 1-based
                    #     # theoretically negatives are allowed in the spec; but hell
                    #     assert (inx > 0), "index should be positive"
                    #     face_indices.append(inx - 1)
                    if len(face_indices) > 0:
                        indices.append(face_indices)
                    if len(face_uv) > 0:
                        f_uv.append(face_uv)
                    if len(face_normals) > 0:
                        f_normals.append(face_normals)
                        
                elif identifier == "vn":
                    assert len(tokens) == 4, f"Expected vertex normals to be 3D in line {line}"
                    normals.append(
                        [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                    )
                elif identifier == "vt":
                    assert len(tokens) == 3, f"Expected UV coordinates to be 2D in line {line}"
                    uvs.append(
                        [float(tokens[1]), float(tokens[2])]
                    )

        if len(uvs) == 0: 
            uvs = None 
        if len(normals) == 0: 
            normals = None 
        if len(f_uv) == 0:
            f_uv = None 
        if len(f_normals) == 0: 
            f_normals = None 
            
        return cls(vertices, indices, uvs, f_uv, normals, f_normals)
        
    @classmethod
    def to_obj(cls, fname, soup):
        raise NotImplementedError()
