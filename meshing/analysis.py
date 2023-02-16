import numpy as np

# Constructs E x 5 matrix of edge neighbors
def computeEdgeNeighborMatrix(mesh):
    edge_matrix = [[edge.index, edge.halfedge.next.edge.index, edge.halfedge.next.next.edge.index,
                        edge.halfedge.twin.next.edge.index, edge.halfedge.twin.next.next.edge.index] 
                   for key, edge in sorted(mesh.topology.edges.items())]
    mesh.edgemat = np.array(edge_matrix, dtype=int)
    
def computeFaceNeighborMatrix(mesh):
    face_matrix = [[face.index, face.halfedge.twin.face.index, face.halfedge.next.twin.face.index,
                    face.halfedge.next.next.twin.face.index] 
                    for key, face in sorted(mesh.topology.faces.items())]
    mesh.facemat = np.array(face_matrix, dtype=int)

def computeFaceToVertex(mesh):
    f_to_v = [] 
    for vind, v in sorted(mesh.topology.vertices.items()):
        f_to_v.append([f.index for f in v.adjacentFaces()])
    mesh.f_to_v = f_to_v # list of lists (potentially different lengths!)

def computeVertexToFace(mesh):
    v_to_f = [[v.index for v in f.adjacentVertices()] for _, f in sorted(mesh.topology.faces.items())]  
    mesh.v_to_f = np.array(v_to_f) # F x 3
    
def computeEdgeToVertex(mesh):
    e_to_v = [] 
    for vind, v in sorted(mesh.topology.vertices.items()):
        e_to_v.append([e.index for e in v.adjacentEdges()])
    mesh.e_to_v = e_to_v
    
def normalize(arr, axis=None, eps=1e-5):
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm[norm < eps] = eps
    return arr / norm

def computeHKS(mesh, k=128, t=np.array([1])):
    from pyhks.hks import get_hks
    vertices, faces, _ = mesh.export_soup() 
    hks = get_hks(vertices, faces, k, t)
    mesh.hks = hks 

def computeFaceNormals(mesh):
    edges1 = []
    edges2 = []
    for key, f in sorted(mesh.topology.faces.items()):
        if f.isBoundaryLoop():
            continue
        u = mesh.vector(f.halfedge)
        v = -1 * mesh.vector(f.halfedge.prev())
        edges1.append(u)
        edges2.append(v)
    n = normalize(np.cross(edges1, edges2, axis=1), axis=1)    
    mesh.facenormals = n

def computeFaceAreas(mesh):
    edges1 = []
    edges2 = []
    for key, f in sorted(mesh.topology.faces.items()):
        if f.isBoundaryLoop():
            continue
        u = mesh.vector(f.halfedge)
        v = -1 * mesh.vector(f.halfedge.prev())
        edges1.append(u)
        edges2.append(v)
    n = 0.5 * np.linalg.norm(np.cross(edges1, edges2, axis=1), axis=1)
    mesh.fareas = n

# Mean of adjacent face normals 
def computeVertexNormals(mesh):
    if not hasattr(mesh, "facenormals"):
        computeFaceNormals(mesh)
    if not hasattr(mesh, "f_to_v"):
        computeFaceToVertex(mesh)
    n = [np.mean(mesh.facenormals[fvec], axis=0) for fvec in mesh.f_to_v]
    mesh.vertexnormals = np.array(n)

# Mean of incident dihedrals  
def computeVertexDihedrals(mesh):
    if not hasattr(mesh, "dihedrals"):
        computeDihedrals(mesh)
    if not hasattr(mesh, "e_to_v"):
        computeEdgeToVertex(mesh)
    vdih = [np.mean(mesh.dihedrals[evec]) for evec in mesh.e_to_v]
    mesh.vertexdihedrals = np.array(vdih)

def computeAngleDeficit(mesh):
    if not hasattr(mesh, "vertexangle"):
        computeVertexAngle(mesh)
    mesh.angledeficit = 2 * np.pi - mesh.vertexangle 
     
def computeVertexEdgeRatios(mesh):
    if not hasattr(mesh, "edgeratios"):
        computeEdgeRatios(mesh)
    if not hasattr(mesh, "e_to_v"):
        computeEdgeToVertex(mesh)
    vedgeratio = [np.mean(mesh.edgeratios[evec,:], axis=0) for evec in mesh.e_to_v]
    mesh.vertexedgeratios = np.array(vedgeratio)
    
def computeDihedrals(mesh):
    if not hasattr(mesh, "facenormals"):
        computeFaceNormals(mesh)
    n1 = []
    n2 = []
    for key, e in sorted(mesh.topology.edges.items()):
        n1.append(mesh.facenormals[e.halfedge.face.index])
        n2.append(mesh.facenormals[e.halfedge.twin.face.index])
    cosTheta = (np.array(n1) * np.array(n2)).sum(axis=1).clip(-1, 1)
    mesh.dihedrals = (np.pi - np.arccos(cosTheta)).reshape(len(cosTheta), 1)

def computeOppositeAngles(mesh):
    v1a = []
    v1b = []
    v2a = []
    v2b = []
    for key, e in sorted(mesh.topology.edges.items()):
        v1a.append(-mesh.vector(e.halfedge.next))
        v1b.append(mesh.vector(e.halfedge.next.next))
        v2a.append(-mesh.vector(e.halfedge.twin.next))
        v2b.append(mesh.vector(e.halfedge.twin.next.next))
    v1a = normalize(np.array(v1a), axis=1)
    v1b = normalize(np.array(v1b), axis=1)
    v2a = normalize(np.array(v2a), axis=1)
    v2b = normalize(np.array(v2b), axis=1)
    cosv1 = (v1a * v1b).sum(axis=1).clip(-1,1)
    cosv2 = (v2a * v2b).sum(axis=1).clip(-1,1)
    angles = np.arccos(np.column_stack((cosv1, cosv2)))
    angles = np.sort(angles, axis=1)
    mesh.symmetricoppositeangles = angles

# Compute interior face angles 
def computeFaceAngles(mesh):
    angles = [] 
    for key, f in sorted(mesh.topology.faces.items()):
        fangle = [] 
        for e in f.adjacentEdges():
            v1 = mesh.vector(e.halfedge)
            v1 /= np.linalg.norm(v1)
            v2 = -mesh.vector(e.halfedge.prev())
            v2 /= np.linalg.norm(v2)
            fangle.append(np.arccos(np.dot(v1, v2).clip(-1,1)))
        angles.append(fangle)
    mesh.faceangles = np.array(angles)
    
def computeVertexAngle(mesh):
    vangles = [] 
    for key, v in sorted(mesh.topology.vertices.items()): 
        angle = 0.0 
        halfedges = list(v.adjacentHalfedges())
        current_vec = mesh.vector(halfedges[0])
        current_vec /= np.linalg.norm(current_vec)
        for next_he in halfedges[1:]:
            next_vec = mesh.vector(next_he)
            next_vec /= np.linalg.norm(next_vec)            
            angle += np.arccos(np.clip(np.dot(current_vec, next_vec), -1, 1))
            current_vec = next_vec
        # Last angle between last and first vectors 
        next_vec = mesh.vector(halfedges[0])
        next_vec /= np.linalg.norm(next_vec)
        angle += np.arccos(np.clip(np.dot(current_vec, next_vec), -1, 1))
        vangles.append(angle)
    mesh.vertexangle = np.array(vangles)

def computeEdgeRatios(mesh):
    # Formula: Area of parallelogram/(edge length)^2
    e1a = []
    e1b = []
    e2a = []
    e2b = []
    lengths = []
    for key, e in sorted(mesh.topology.edges.items()):
        e1a.append(mesh.vector(e.halfedge))
        e1b.append(-1 * mesh.vector(e.halfedge.prev()))
        e2a.append(mesh.vector(e.halfedge.twin))
        e2b.append(-1 * mesh.vector(e.halfedge.twin.prev()))
        lengths.append(np.linalg.norm(mesh.vector(e.halfedge)))
    ratio1 = np.linalg.norm(np.cross(e1a, e1b, axis=1), axis=1)/np.array(lengths)**2
    ratio2 = np.linalg.norm(np.cross(e2a, e2b, axis=1), axis=1)/np.array(lengths)**2
    ratios = np.column_stack((ratio1, ratio2))
    ratios = np.sort(ratios, axis=1)
    mesh.edgeratios = ratios