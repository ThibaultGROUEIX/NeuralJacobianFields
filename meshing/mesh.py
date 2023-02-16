import numpy as np
from numpy import cross, dot
from numpy.linalg import norm
from . import Halfedge, Edge, Vertex, Face, Topology
from .analysis import computeVertexDihedrals, computeVertexEdgeRatios, computeAngleDeficit, normalize
from .analysis import computeDihedrals, computeOppositeAngles, computeEdgeRatios
from pathlib import Path
from source_njf.utils import fix_orientation 
"""
every element index is unique and never recycled
vertex deletion does not modify the self.vertices array.
It's only the topology that is modified
"""

class Mesh:
    def __init__(self, vertices=None, face_indices=None, uvs=None, meshdata=None, meshname=None, export_dir=None):
        self.topology = Topology()
        self.meshname=meshname
        self.export_dir = export_dir
        self.anchor_fs = None 
        if meshdata is not None:                        
            self.topology = self.topology.build_from_halfedge_serialization(meshdata['halfedge_data'], do_check=True)
            for attr in meshdata.keys(): 
                setattr(self, attr, meshdata[attr])
        else:
            self.vertices = vertices
            self.faces = face_indices.astype(int)
            self.uvs = uvs 
            # self.faces = fix_orientation(self.vertices, face_indices.astype(int))
            self.topology.build(len(vertices), self.faces)

    def export_soup(self):
        init_n = len(self.vertices)
        face_conn = np.array(self.topology.export_face_connectivity(), dtype=np.uint32)
        edge_conn = np.array(self.topology.export_edge_connectivity(), dtype=np.uint32)

        # only export remaining vertices; hence we never have to modify the
        # numpy array during edits
        # is the code below readable?
        old_inds = np.array(sorted(self.topology.vertices.keys()))
        new_inds = np.arange(len(old_inds), dtype=np.int)
        vertices = self.vertices[old_inds]
        A = np.zeros(init_n, dtype=np.int64)
        A[old_inds] = new_inds

        face_conn = A[face_conn]
        edge_conn = A[edge_conn]
        return vertices, face_conn, edge_conn

    def export_obj(self, export_dir=None, meshname=None, uv=None, fuv = None, vnormals=None, fnormals=None):
        import os
        if not self.export_dir and not export_dir:
            return
        if not meshname and not self.meshname:
            return 
        if not export_dir: 
            export_dir = self.export_dir
        if not meshname: 
            meshname = self.meshname
        Path(export_dir).mkdir(exist_ok=True, parents=True)
        
        file = os.path.join(export_dir, f"{meshname}.obj")
        vertices, faces, edges = self.export_soup()
        with open(file, 'w') as f:
            for vi, v in enumerate(vertices):
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            if uv is not None: 
                for v_uv in uv: 
                    f.write(f"vt {v_uv[0]} {v_uv[1]} \n")
            if vnormals is not None: 
                for vnormal in vnormals: 
                    f.write(f"vn {vnormal[0]} {vnormal[1]} {vnormal[2]}\n")             
            if fuv is not None: 
                if fnormals is not None: 
                    for i in range(len(faces)):
                        face = faces[i] 
                        faceuv = fuv[i]
                        fnormal = fnormals[i]
                        f.write(f"f {face[0]+1:d}/{faceuv[0]+1:d}/{fnormal[0]+1:d} {face[1]+1:d}/{faceuv[1]+1:d}/{fnormal[1]+1:d} {face[2]+1:d}/{faceuv[2]+1:d}/{fnormal[2]+1:d}\n") 
                else:             
                    for i in range(len(faces)):
                        face = faces[i] 
                        faceuv = fuv[i]
                        f.write(f"f {face[0]+1:d}/{faceuv[0]+1:d} {face[1]+1:d}/{faceuv[1]+1:d} {face[2]+1:d}/{faceuv[2]+1:d}\n")   
            elif fnormals is not None:
                for i in range(len(faces)):
                    face = faces[i] 
                    fnormal = fnormals[i]
                    f.write(f"f {face[0]+1:d}//{fnormal[0]+1:d} {face[1]+1:d}//{fnormal[1]+1:d} {face[2]+1:d}//{fnormal[2]+1:d}\n")
            else:
                for i in range(len(faces)):
                    face = faces[i] 
                    f.write(f"f {face[0]+1:d} {face[1]+1:d} {face[2]+1:d}\n")
            for edge in edges[:-1]:
                f.write("e %d %d\n" % (edge[0] + 1, edge[1] + 1))
            f.write("e %d %d\n" % (edges[-1][0] + 1, edges[-1][1] + 1))
            
    # Return vertices and face connectivities associated with input faces 
    def export_submesh(self, face_inds):
        init_n = len(self.vertices)
        
        face_conn = np.array(self.topology.export_face_connectivity(), dtype=np.uint32)
        face_conn = face_conn[face_inds]
        v_inds = np.sort(np.unique(face_conn))
        vertices = self.vertices[v_inds]
        A = np.zeros(init_n, dtype=np.int64)
        A[v_inds] = np.arange(len(v_inds))
        face_conn = A[face_conn]
        
        # Edge case: exactly one face chosen 
        if len(face_conn.shape) == 1: 
            face_conn = face_conn.reshape(1,3)
        
        return vertices, face_conn

    def normalize(self, copy_v=False):
        if copy_v:
            import copy 
            newverts = copy.copy(self.vertices)
            newverts -= np.mean(newverts, axis=0)
            newverts /= np.max(np.linalg.norm(newverts, axis=1))
            self.vertices = newverts
        else:
            self.vertices -= np.mean(self.vertices, axis=0)
            self.vertices /= np.max(np.linalg.norm(self.vertices, axis=1))

    def get_3d_pos(self, v: Vertex):
        return self.vertices[v.index]

    def vector(self, h: Halfedge):
        a = self.get_3d_pos(h.vertex)
        b = self.get_3d_pos(h.next.vertex)
        return (b - a)

    def length(self, e: Edge):
        return norm(self.vector(e.halfedge))

    def midpoint(self, e: Edge):
        h = e.halfedge
        a = self.get_3d_pos(h.vertex)
        b = self.get_3d_pos(h.next.vertex)
        p = (a + b) / 2
        return p

    def meanEdgeLength(self):
        sum = 0
        edges = self.topology.edges
        for e in edges:
            sum += self.length(e)
        mean = sum / len(edges)
        return mean

    def area(self, f: Face):
        """area of a face"""
        if f.isBoundaryLoop():
            return 0.

        u = self.vector(f.halfedge)
        v = -1 * self.vector(f.halfedge.prev())
        area = 0.5 * norm(cross(u, v))
        return area

    def totalArea(self):
        sum = 0.0
        for f in self.topology.faces.values():
            sum += self.area(f)
        return sum

    def faceNormal(self, f: Face):
        if f.isBoundaryLoop():
            return None
        u = self.vector(f.halfedge)
        v = -1 * self.vector(f.halfedge.prev())
        n = normalize(cross(u, v))
        return n

    def vertexNormal(self, v: Vertex):
        norm = 0
        i = 0
        for f in v.adjacentFaces():
            norm += self.faceNormal(f)
            i += 1
        norm /= i
        return norm

    def angle(self, h: Halfedge):
        """angle across a halfedge; or angle at a corner"""
        u = normalize(self.vector(h.prev()))
        v = normalize(-1 * self.vector(h.next))
        ang = np.arccos(max(-1.0, min(1.0, u.dot(v))))
        return ang

    def dihedralAngle(self, h: Halfedge):
        if h.onBoundary or h.twin.onBoundary:
            return 0.0
        n1 = self.faceNormal(h.face)
        n2 = self.faceNormal(h.twin.face)
        cosTheta = n1.dot(n2)
        angle = np.pi - np.arccos(cosTheta)
        return angle

    def computeEdgeFeatures(self, overwrite=False, intrinsics=[]):
        featuredict = {"dihedrals": computeDihedrals, "symmetricoppositeangles": computeOppositeAngles, "edgeratios":computeEdgeRatios}
        if not hasattr(self, "edgefeatures") or overwrite: 
            with np.errstate(divide='raise'):
                try:
                    edgefeatures = [] 
                    for feature in featuredict.keys():
                        if not hasattr(self, feature) or overwrite:
                            featuredict[feature](self)
                        if feature in intrinsics:
                            edgefeature = getattr(self, feature)
                            edgefeatures.append(edgefeature.transpose(1,0))
                    if len(edgefeatures) == 0:
                        self.edgefeatures = np.array([])
                    else:
                        self.edgefeatures = np.vstack(edgefeatures) # F x E
                except Exception as e:
                    print(e)
                    raise ValueError(self.meshname, 'bad features') 
    
    def computeVertexFeatures(self, overwrite=False, intrinsics=[]):
        featuredict = {"vertexdihedrals": computeVertexDihedrals, "angledeficit": computeAngleDeficit, "vertexedgeratios":computeVertexEdgeRatios}
        if not hasattr(self, "vertexfeatures") or overwrite: 
            with np.errstate(divide='raise'):
                try:
                    vertexfeatures = [] 
                    for feature in featuredict.keys():
                        if not hasattr(self, feature) or overwrite:
                            featuredict[feature](self)
                        if feature in intrinsics:
                            vertexfeatures.append(getattr(self, feature))
                    if len(vertexfeatures) == 0:
                        self.vertexfeatures = np.array([])
                    else:
                        self.vertexfeatures = np.column_stack(vertexfeatures) # V x C 
                except Exception as e:
                    print(e)
                    raise ValueError(self.meshname, 'bad features') 
