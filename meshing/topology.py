import numpy as np
from collections import defaultdict
from . import Halfedge, Vertex, Edge, Face

# For an interior face, the vertices go in CCW order according to Right Hand Rule
# For an exterior face, the vertices go in CW order
# a halfedge stores its source vertex


class ElemCollection(dict):
    """This dict wrapper keeps track of the number of uniquly allocated elements
    so that each element has an unambiguous id in the lifetime of the mesh (even after edits)
    """
    def __init__(self, cons_f):
        super().__init__()
        self.cons_f = cons_f
        self.n_allocations = 0

    def allocate(self):
        elem = self.cons_f()
        i = self.n_allocations
        elem.index = i
        self[i] = elem
        self.n_allocations += 1
        return elem

    def fill_vacant(self, elem_id):
        """an element was previously deleted, and is now re-inserted"""
        assert elem_id not in self
        elem = self.cons_f()
        elem.index = elem_id
        self[elem_id] = elem
        return elem

    def compactify_keys(self):
        """fill the holes in index keys"""
        store = dict()
        for i, (_, elem) in enumerate(sorted(self.items())):
            store[i] = elem
            elem.index = i
        self.clear()
        self.update(store)


class Topology():
    def __init__(self):
        self.halfedges = ElemCollection(Halfedge)
        self.vertices = ElemCollection(Vertex)
        self.edges = ElemCollection(Edge)
        self.faces = ElemCollection(Face)
        self.boundaries = ElemCollection(Face)

    def eulerCharacteristic(self):
        """Compute the Euler Characteristic of the Mesh. Useful to compute Genus
        """
        return len(self.vertices) - len(self.edges) + len(self.faces)

    def genus(self):
        chi = self.eulerCharacteristic()
        nBoundaryLoops = len(self.boundaries)
        g = (2 - nBoundaryLoops - chi) // 2
        return g

    def build(self, n_vertices, indices) -> bool:
        """
        returns True if this mesh is constructed successfully
        returns False if any of the following is true:
            - non-manifold vertices
            - non-manifold edges
            - isolated vertices
            - isolated faces
        """
        indices = indices.reshape(-1)

        # create and insert vertices
        indexToVertex = dict()
        for i in range(n_vertices):
            v = self.vertices.allocate()
            indexToVertex[i] = v

        # create and insert halfedges, edges and non boundary loop faces
        edgeCount = dict()  # number of halfedges each edge has; used to track non-manifold edges
        existingHalfedges = dict()
        hasTwinHalfedge = dict()
        for I in range(0, len(indices), 3):
            # create new face
            f = self.faces.allocate()

            # create a halfedge for each edge of the newly created face
            for J in range(3):
                h = self.halfedges.allocate()

            # initialize the newly created halfedges
            for J in range(3):
                # // current halfedge goes from vertex i to vertex j
                K = (J + 1) % 3
                i = indices[I + J]
                j = indices[I + K]

                # set the current halfedge's attributes
                h = self.halfedges[I + J]
                h.next = self.halfedges[I + K]
                h.onBoundary = False
                hasTwinHalfedge[h] = False

                # point the new halfedge and vertex i to each other
                v = indexToVertex[i]
                h.vertex = v
                v.halfedge = h # NOTE: this vertex object is accessed multiple times according to its degree (the halfedge it receives is the LAST one processed)

                # point the new halfedge and face to each other
                h.face = f
                f.halfedge = h

                # swap if i > j
                key = (i, j) if i <= j else (j, i)
                if key in existingHalfedges:
                    # if a halfedge between vertex i and j has been created in the past, then it
                    # is the twin halfedge of the current halfedge
                    twin = existingHalfedges[key]
                    h.twin = twin
                    twin.twin = h
                    h.edge = twin.edge

                    hasTwinHalfedge[h] = True
                    hasTwinHalfedge[twin] = True
                    edgeCount[key] += 1
                else:
                    # create an edge and set its halfedge
                    e = self.edges.allocate()
                    h.edge = e
                    e.halfedge = h

                    # record the newly created edge and halfedge from vertex i to j
                    existingHalfedges[key] = h
                    edgeCount[key] = 1

                # check for non-manifold edges
                if edgeCount[key] > 2:
                    print(f"Mesh has non-manifold edges: {key}")
                    return False

        # create and insert boundary halfedges and "imaginary" faces for boundary cycles
        # also create and insert corners

        # the number of halfedges created so far is len(indices) or 3 x num_faces
        # we go from here
        for i in range(0, len(indices)):
            # if a halfedge has no twin halfedge, create a new face and
            # link it to the corresponding boundary cycle
            h = self.halfedges[i]
            if not hasTwinHalfedge[h]:
                # create new face
                f = self.boundaries.allocate()

                # walk along boundary cycle
                boundaryCycle = []
                he = h
                while True:
                    # create a new halfedge
                    bH = self.halfedges.allocate()
                    boundaryCycle.append(bH)

                    # grab the next halfedge along the boundary that does not have a twin halfedge
                    nextHe = he.next
                    while hasTwinHalfedge[nextHe]:
                        nextHe = nextHe.twin.next

                    # set the current halfedge's attributes
                    bH.vertex = nextHe.vertex  # boundary halfedges go CW; i.e. backward
                    bH.edge = he.edge
                    bH.onBoundary = True

                    # point the new halfedge and face to each other
                    bH.face = f
                    f.halfedge = bH

                    # point the new halfedge and he to each other
                    bH.twin = he
                    he.twin = bH

                    # continue walk
                    he = nextHe

                    if (he == h):
                        break

                # link the cycle of boundary halfedges together
                n = len(boundaryCycle)
                for j in range(n):
                    # boundary halfedges are linked in clockwise order
                    boundaryCycle[j].next = boundaryCycle[(j + n - 1) % n]
                    hasTwinHalfedge[boundaryCycle[j]] = True
                    hasTwinHalfedge[boundaryCycle[j].twin] = True

            # We are not creating corner at the moment

        # check if mesh has isolated vertices, isolated faces or non-manifold vertices
        if self.hasIsolatedVertices() or self.hasIsolatedFaces() or self.hasNonManifoldVertices():
            return False

        return True

    def compactify_keys(self):
        self.halfedges.compactify_keys()
        self.vertices.compactify_keys()
        self.edges.compactify_keys()
        self.faces.compactify_keys()
        self.boundaries.compactify_keys()

    def export_halfedge_serialization(self):
        """
        this provides the unique, unambiguous serialization of the halfedge topology
        i.e. one can reconstruct the mesh connectivity from this information alone
        It can be used to track history, etc.
        """
        data = []
        for _, he in sorted(self.halfedges.items()):
            data.append(he.serialize())
        data = np.array(data, dtype=np.int32)
        return data

    @classmethod
    def build_from_halfedge_serialization(cls, halfedge_data, do_check=False):
        n_halfedges = halfedge_data[:, 0].max() + 1
        n_verts = halfedge_data[:, 1].max() + 1
        n_edges = halfedge_data[:, 2].max() + 1
        on_bound = halfedge_data[:, -1].astype(np.bool)
        n_faces = 0 if on_bound.all() else halfedge_data[~on_bound][:, 3].max() + 1
        n_bounds = 0 if not on_bound.any() else halfedge_data[on_bound][:, 3].max() + 1

        def allocate_n_elements(collection: ElemCollection, n):
            for i in range(n):
                collection.allocate()

        topo = Topology()
        allocate_n_elements(topo.halfedges, n_halfedges)
        allocate_n_elements(topo.vertices, n_verts)
        allocate_n_elements(topo.edges, n_edges)
        allocate_n_elements(topo.faces, n_faces)
        allocate_n_elements(topo.boundaries, n_bounds)

        for i in range(len(halfedge_data)):
            i_he, i_vert, i_edge, i_face, i_next, i_twin, is_bound = halfedge_data[i].tolist()
            is_bound = bool(is_bound)
            # assert i == i_he
            he = topo.halfedges[i_he]
            vert = topo.vertices[i_vert]
            edge = topo.edges[i_edge]
            face = topo.faces[i_face] if not is_bound else topo.boundaries[i_face]
            he_next = topo.halfedges[i_next]
            he_twin = topo.halfedges[i_twin]

            he.vertex = vert
            he.edge = edge
            he.face = face
            he.next = he_next
            he.twin = he_twin
            he.onBoundary = is_bound
            vert.halfedge = he
            edge.halfedge = he
            face.halfedge = he

        # if do_check:
        #     topo.thorough_check()
        return topo

    def export_face_connectivity(self):
        face_indices = []
        for inx, face in sorted(self.faces.items()):
            vs_of_this_face = []
            if face.halfedge is None:
                continue
            for vtx in face.adjacentVertices():
                vs_of_this_face.append(vtx.index)
            assert len(vs_of_this_face) == 3
            face_indices.append(vs_of_this_face)
        return face_indices

    def export_edge_connectivity(self):
        conn = []
        for _, edge in sorted(self.edges.items()):
            if edge.halfedge is None:
                continue
            v1 = edge.halfedge.vertex
            v2 = edge.halfedge.twin.vertex
            conn.append([v1.index, v2.index])
        return conn

    def export_edge_face_connectivity(self, fs):
        # fconn: E x 2 indexed by neighboring faces
        # vconn: E x 2 x 2 indexed by local vertex indices of neighboring indices ([v0, v1, v0', v1'])
        # NOTE: we have to take faces as input because topology face to vertex ordering is arbitrary
        fconn = []
        vconn = []
        for _, edge in sorted(self.edges.items()):
            if edge.halfedge is None:
                continue

            # Don't count edges on a boundary
            if edge.onBoundary():
                continue

            fconn.append([edge.halfedge.face.index, edge.halfedge.twin.face.index])

            # Find vertex indices corresponding to each face
            ev0 = np.array([-1, -1])
            ev1 = np.array([-1, -1])
            count = 0
            v0 = edge.halfedge.vertex.index
            v1 = edge.halfedge.twin.vertex.index
            f0 = edge.halfedge.face.index
            for v in fs[f0]:
                if v == v0:
                    ev0[0] = count
                elif v == v1:
                    ev0[1] = count
                if np.all(ev0 >= 0):
                    break
                count += 1


            count = 0
            f1 = edge.halfedge.twin.face.index
            for v in fs[f1]:
                if v == v0:
                    ev1[0] = count
                elif v == v1:
                    ev1[1] = count
                if np.all(ev1 >= 0):
                    break
                count += 1
            assert np.all(np.array(ev0) >= 0), f"Face {edge.halfedge.face.index} missing corresponding vertex for edge {edge.index}"
            assert np.all(np.array(ev1) >= 0), f"Face {edge.halfedge.twin.face.index} missing corresponding vertex for edge {edge.index}"
            assert np.all(np.array(ev0) <= 2), f"Face {edge.halfedge.face.index} invalid correspondence vertices {ev0}"
            assert np.all(np.array(ev1) <= 2), f"Face {edge.halfedge.twin.face.index} invalid correspondence vertices {ev1}"

            vconn.append([ev0, ev1])
        return fconn, vconn

    def hasIsolatedVertices(self):
        for v in self.vertices.values():
            if v.isIsolated():
                print("Mesh has isolated vertices")
                return True
        return False

    def hasIsolatedFaces(self):
        for f in self.faces.values():
            boundaryEdges = 0
            for h in f.adjacentHalfedges():
                if h.twin.onBoundary:
                    boundaryEdges += 1
            if (boundaryEdges == 3):
                print("Mesh has isolated faces!")
                return True

        return False

    def hasNonManifoldVertices(self):
        # number of adjacent faces can be computed exactly
        # define each linked set of edges a bundle. Vertex degree will only
        # capture one bundle
        # hence vertex degree != num of adjacent faces
        adjacentFaces = dict()
        for v in self.vertices.values():
            adjacentFaces[v] = 0

        for f in self.faces.values():
            for v in f.adjacentVertices():
                adjacentFaces[v] += 1

        for b in self.boundaries.values():
            for v in b.adjacentVertices():
                adjacentFaces[v] += 1

        self.nonmanifvs = []
        for v in self.vertices.values():
            if adjacentFaces[v] != v.degree():
                self.nonmanifvs.append(v.index)

        if len(self.nonmanifvs) > 0:
            return True

        del adjacentFaces
        return False

    def hasNonManifoldEdges(self):
        # Each edge should be associated with exactly 2 half-edges (incl. boundary)
        edgeCounts = defaultdict(int)
        for he in self.halfedges.values():
            edgeCounts[he.edge] += 1
            if edgeCounts[he.edge] > 2:
                return True
        return False

    def computeNonManifoldEdges(self):
        edgeCounts = defaultdict(int)
        nonmanifcount = 0
        for he in self.halfedges.values():
            edgeCounts[he.edge] += 1
            if edgeCounts[he.edge] > 2:
                nonmanifcount += 1
        return nonmanifcount

    def thorough_check(self):
        # a halfedge has v, e, f, prev, next, twin, onBoundary
        # check halfedge through major elements

        # foreach edge: e has 2 he. he: twin; he: edge; cover all;
        # foreach vertex: v has k hes; check he: v; # cover all  # cover all

        # foreach face or bound:
        # f has k hes through next iteration; he: next, he: prev, he: face he: onBoundary; cover all

        def check_indexing(src_dict):
            for inx, v in src_dict.items():
                assert inx == v.index

        check_indexing(self.halfedges)
        check_indexing(self.vertices)
        check_indexing(self.edges)
        check_indexing(self.faces)
        check_indexing(self.boundaries)

        # 2. check edges
        self._check_verts()

        self._check_edges()
        self._check_faces_and_boundary_loops()

        assert not self.hasIsolatedVertices()
        assert not self.hasIsolatedFaces()
        assert not self.hasNonManifoldVertices()
        assert not self.hasNonManifoldEdges()

        # TODO: Delete this
        # return True, True, True

    def _check_verts(self):
        encountered_halfedges = []
        for inx, v in self.vertices.items():
            hes = []
            for he in v.adjacentHalfedges():
                # if he.vertex != v:
                #     return False, he, v
                assert he.vertex == v
                hes.append(he)
            encountered_halfedges.extend([elem.index for elem in hes])
        encountered_halfedges = set(encountered_halfedges)
        assert encountered_halfedges == set(self.halfedges.keys()), "must cover all halfedges"
        # return True, True, True
    def _check_edges(self):
        encountered_halfedges = []
        for inx, e in self.edges.items():
            he = e.halfedge
            twin = he.twin

            hes = [he, twin]
            n = len(hes)

            for i, he in enumerate(hes):
                assert he.edge == e
                assert he.twin == hes[(i + 1) % n]

            encountered_halfedges.extend([elem.index for elem in hes])

        encountered_halfedges = set(encountered_halfedges)
        assert encountered_halfedges == set(self.halfedges.keys()), "must cover all halfedges"

    def _check_faces_and_boundary_loops(self):
        def inner_routine(set_of_faces, onBound: bool):
            encountered_halfedges = []
            for inx, f in set_of_faces.items():
                hes = []
                for he in f.adjacentHalfedges():
                    hes.append(he)

                if not onBound:  # non boundary loops can be >= 3
                    assert len(hes) == 3

                n = len(hes)
                for i, he in enumerate(hes):
                    assert he.face == f
                    assert he.onBoundary is onBound
                    assert he.next == hes[(i + 1) % n]

                encountered_halfedges.extend([elem.index for elem in hes])

            encountered_halfedges = set(encountered_halfedges)
            target_halfedges = {
                k for k, v in self.halfedges.items()
                if v.onBoundary is onBound
            }
            assert encountered_halfedges == target_halfedges, \
                f"should cover all he whose onBoundary is {onBound}"

        inner_routine(self.faces, False)
        inner_routine(self.boundaries, True)

    def produce_edge_mask(self, id_set):
        """this generates a binary mask for edges is the id_set
        visualization purpose only
        """
        mask = [
            1 if e_id in id_set else 0
            for e_id, _ in self.edges.items()
        ]
        return mask
