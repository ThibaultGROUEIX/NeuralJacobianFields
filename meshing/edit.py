from . import Halfedge, Edge, Vertex, Face, Topology, Mesh
import numpy as np


class MeshEdit():
    def __init__(self):
        pass

    def apply(self):
        raise NotImplementedError()

    def inverse(self):
        raise NotImplementedError()

# Laplacian smoothing with custom vertex weights (basically how much of original vertex position to keep)
class LaplacianSmoothing(MeshEdit):
    def __init__(self, mesh, n=1, weights=None):
        self.mesh = mesh
        self.n = n
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(len(self.mesh.vertices))

    def apply(self):
        # Build sparse laplacian matrix
        from scipy.sparse import csr_matrix
        dense_lapl = np.diag(self.weights)
        lapl_vs = [v.index for _, v in sorted(self.mesh.topology.vertices.items()) for v_n in v.adjacentVertices()]
        lapl_counts = []
        lapl_ns = []
        for _, v in sorted(self.mesh.topology.vertices.items()):
            neighbors = [v_n.index for v_n in v.adjacentVertices()]
            lapl_counts.append(len(neighbors))
            lapl_ns.extend(neighbors)
        lapl_weights = []
        for i in range(len(self.weights)):
            lapl_weights.extend([(1 - self.weights[i])/lapl_counts[i]] * lapl_counts[i])
        dense_lapl[lapl_vs, lapl_ns] = lapl_weights
        dense_lapl /= np.sum(dense_lapl, axis=1, keepdims=True)
        sparse_lapl = csr_matrix(dense_lapl)
        for _ in range(self.n):
            self.mesh.vertices = sparse_lapl @ self.mesh.vertices

class EdgeFlip(MeshEdit):
    def __init__(self, mesh, e_id):
        self.mesh = mesh
        self.e_id = e_id

    def apply(self, preventSelfEdges=True):
        # TODO: should only care about triangles; polygon flips are even messier
        # with more edge cases
        e = self.mesh.topology.edges[self.e_id]
        if e.onBoundary():
            # or return None for rejection
            # raise ValueError("edge is on boundary; cannot flip edges")
            return None

        # this routine is written with allowing polygonal faces in mind.
        # when it's triangular, h3 is h0
        h1 = e.halfedge
        h2 = h1.next
        h3 = h2.next
        h0 = h1.prev()

        h5 = h1.twin
        h6 = h5.next
        h7 = h6.next
        h4 = h5.prev()

        f0 = h1.face
        f1 = h5.face

        v1 = h1.vertex
        v2 = h7.vertex
        v4 = h5.vertex
        v5 = h3.vertex

        if preventSelfEdges:
            for _v in v2.adjacentVertices():
                if _v == v5:
                    # raise ValueError("self edge")
                    return None

        # halfedge's next
        h0.next = h6
        h1.next = h3
        h2.next = h5
        # h3.next no change

        h4.next = h2
        h5.next = h7
        h6.next = h1
        # h7.next no change

        # halfedge's twin
        # halfedge's vertex
        h1.vertex = v2
        h5.vertex = v5

        # halfedge's face
        h2.face = f1
        h6.face = f0

        # vertex's halfedges
        v1.halfedge = h6
        v4.halfedge = h2

        # face's halfedges
        f0.halfedge = h1
        f1.halfedge = h5

        # edge's halfedges
        return e

    def inverse(self):
        # assume that we only have triangular mesh, and so 1 more flip suffices to undo the flip
        return EdgeFlip(self.mesh, self.e_id)

class EdgeCollapse(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        # case: e is a boundary edge; reject; this is doable but not for now
        # case: both verts at an edge is on boundary, reject
        # case: if either verts of the edge is on boundary, reject too
        # case: pinch triangle
        # case: valence > 3 or valence 3 collapse
        # TODO: link condition
        event = do_collapse(mesh, e_id)
        self.do_able, self.record = next(event)
        if self.do_able is False:
            return
        self.event = event

    def apply(self):
        return next(self.event)

    def inverse(self):
        # Record every deleted mesh element for true inverse map
        v_top_id, v_bottom_id, e_left_id, e_right_id, v_top_coord, v_bottom_coord, \
        new_e_bundle, new_e_left_bundle, new_e_right_bundle, new_f_bundle  = self.record
        return VertexSplit(self.mesh, v_top_id, e_left_id, e_right_id, v_top_coord, v_bottom_coord, v_bottom_id,
                           new_e_bundle, new_e_left_bundle, new_e_right_bundle, new_f_bundle)

def do_collapse(mesh, e_id):
    topology = mesh.topology
    # Not allowed to collapse past tetrahedron
    if len(topology.edges.keys()) <= 6:
        yield False, ()
    if e_id not in topology.edges.keys():
        yield False, ()
    e = topology.edges[e_id]

    if e.onBoundary():
        yield False, ()
    heA0 = e.halfedge # NOTE: we delete the vertex assigned to this halfedge
    if heA0.vertex.onBoundary() or heA0.twin.vertex.onBoundary():
        yield False, ()

    # check "pinch" triangle?
    # the half-flaps of an edge connect its two verts with a pair of len=2 walks
    # however there can be more than just this pair of len=2 walks.
    # i.e the edge is "pinched" by another pair. see diagram below
    # https://stackoverflow.com/questions/27049163/mesh-simplification-edge-collapse-conditions/27049418
    # in this case, collapsing this edge creates
    # 1) delta complex: two verts linked by more than 1 edge
    # 2) zero dihedral-angle triangle pairs
        """ a case of pinch triangles
            *
           * * -
          *   *    -
        * * * * * - - -
          *   *    -
           * * -
            *
        """
    for he1 in heA0.tip_vertex().adjacentHalfedges():
        for he2 in he1.tip_vertex().adjacentHalfedges():
            if (
                not (heA0.next == he1 and he1.next == he2) and
                not (he1.twin.next.twin == heA0 and he2.twin.next.twin == he1)
            ):
                # not just going around a face
                if (he2.tip_vertex() == heA0.vertex):
                    yield False, ()

    def delete_edge_bundle(e):
        heA = e.halfedge
        heB = heA.twin
        del topology.halfedges[heA.index]
        del topology.halfedges[heB.index]
        del topology.edges[e.index]

    if heA0.vertex.degree() > 3:
        heA1 = heA0.next
        heA2 = heA1.next
        heB0 = heA0.twin
        heB1 = heB0.next
        heB2 = heB1.next
        heC2 = heA2.twin
        heC0 = heC2.next
        heC1 = heC0.next
        heD1 = heB1.twin
        heD2 = heD1.next
        heD0 = heD2.next

        fA = heA0.face
        fB = heB0.face
        fC = heC0.face
        fD = heD0.face

        vA = heA0.vertex
        vB = heB0.vertex
        vC = heC0.vertex
        vD = heD1.vertex

        # TODO: Save deleted edge bundles as well
        # Order of edge bundles: primary, left, right (orientation is deleted vertex on bottom, combined vertex is top)
        record = (
            vB.index, vA.index, heA1.edge.index, heB2.edge.index,
            mesh.vertices[vB.index].copy(), mesh.vertices[vA.index].copy(),
            (e_id, e.halfedge.index, e.halfedge.twin.index),
            (heA2.edge.index, heA2.index, heA2.twin.index),
            (heB1.edge.index, heB1.index, heB1.twin.index),
            (fA.index, fB.index)
        )
        yield True, record

        for he in vA.adjacentHalfedges():
            he.vertex = vB

        heD0.next = heB2
        heB2.next = heD2
        heC1.next = heA1
        heA1.next = heC0
        heB2.face = fD
        heA1.face = fC
        fC.halfedge = heC0
        fD.halfedge = heD0

        vB.halfedge = heA1
        vC.halfedge = heC0
        vD.halfedge = heB2

        Vertex.merge_into(mesh, vA, vB)

        delete_edge_bundle(e)
        delete_edge_bundle(heB1.edge)
        delete_edge_bundle(heA2.edge)
        del topology.vertices[vA.index]
        del topology.faces[fA.index]
        del topology.faces[fB.index]

        yield vB

    elif heA0.vertex.degree() == 3:
        heA1 = heA0.next
        heA2 = heA1.next
        heB0 = heA0.twin
        heB1 = heB0.next
        heB2 = heB1.next
        heC2 = heA2.twin
        heC0 = heC2.next
        heC1 = heC0.next

        fA = heA0.face
        fB = heB0.face
        fC = heC0.face

        vA = heA0.vertex
        vB = heB0.vertex
        vC = heC0.vertex
        vD = heC1.vertex

        record = (
            vB.index, vA.index, heA1.edge.index, heB2.edge.index,
            mesh.vertices[vB.index].copy(), mesh.vertices[vA.index].copy(),
            (e_id, e.halfedge.index, e.halfedge.twin.index),
            (heA2.edge.index, heA2.index, heA2.twin.index),
            (heB1.edge.index, heB1.index, heB1.twin.index),
            (fA.index, fB.index)
        )
        yield True, record

        heA1.next = heC0
        heC0.next = heB2
        heB2.next = heA1
        heB2.face = fC
        heA1.face = fC
        fC.halfedge = heC0

        vB.halfedge = heA1
        vC.halfedge = heC0
        vD.halfedge = heB2

        Vertex.merge_into(mesh, vA, vB)

        delete_edge_bundle(e)
        delete_edge_bundle(heB1.edge)
        delete_edge_bundle(heA2.edge)
        del topology.vertices[vA.index]
        del topology.faces[fA.index]
        del topology.faces[fB.index]

        yield vB
    else:
        yield False, ()

# class EdgeCut():
#     def __init__(
#         self, mesh, he_i
#     ):
#         self.mesh = mesh
#         self.he_i = he_i

#     def apply(self):
#         he = self.mesh.topology.halfedges[self.he_i]
#         sourcev = he.vertex
#         targetv = he.tip_vertex()

#         # Cut: 1 new vertex, 1 new edge, two new halfedges
#         # New edge -> one new halfedge
#         # New halfedges -> next each other, twin with cut faces, 1 old vertex, other next vertex, split faces, 1 new edge, 1 old edge
#         # New vertex -> assign to hes up to cut edge
#         newv = self.mesh.topology.vertices.allocate()
#         newe = self.mesh.topology.edges.allocate()
#         newh1 = self.mesh.topology.halfedges.allocate()
#         newh2 = self.mesh.topology.halfedges.allocate()

#         # New vertex position
#         self.mesh.vertices = np.concatenate([self.mesh.vertices, np.array([self.mesh.vertices[sourcev.index]])], axis=0)

#         # NOTE: We assume the cut starts from a vertex on the boundary!!!
#         # Start the sweep from the boundary halfedge
#         he_bounds = [he for he in sourcev.adjacentHalfedges() if he.onBoundary]
#         assert len(he_bounds) == 1, f"Source vertex has {len(he_bounds)} boundaries."

#         # Assign primitives
#         newv.halfedge = he
#         sourcev.halfedge = newh1
#         newh1.vertex = sourcev
#         sourcev.halfedge = newh1
#         newh1.onBoundary = True
#         newh1.next = newh2
#         newh1.twin = he.twin
#         he.twin.twin = newh1
#         newh1.face = he.twin.face
#         newh1.edge = he.edge
#         he.edge.halfedge = newh1
#         he_bounds[0].prev().next = newh1 # This connects the new he to the boundary

#         # Assign new edge
#         newe.halfedge = newh2
#         newh2.edge = newe
#         newh2.vertex = targetv
#         targetv.halfedge = newh2
#         newh2.onBoundary = True
#         newh2.next = he_bounds[0]
#         newh2.twin = he
#         he.twin = newh2
#         newh2.face = he.face
#         he.edge = newe

#         # Edge case: if target vertex on two boundaries, then we need to split target vertex and attach to the new boundary
#         targetbhe = [he for he in targetv.adjacentHalfedges() if he.onBoundary and he != newh2]
#         if len(targetbhe) > 0:
#             assert len(targetbhe) == 1, f"Target vertex has {len(targetbhe)} boundaries."
#             newtargetv = self.mesh.topology.vertices.allocate()
#             self.mesh.vertices = np.concatenate([self.mesh.vertices, np.array([self.mesh.vertices[targetv.index]])], axis=0)

#             # Reassign topology
#             prevhe = targetbhe[0].prev()
#             newh1.next = targetbhe[0]
#             prevhe.next = newh2

#             # Reassign fan from newh2 to prevhe to new vertex
#             newh2.vertex = newtargetv
#             tmphe = newh2.twin.next
#             while tmphe != newh2:
#                 tmphe.vertex = newtargetv
#                 tmphe = tmphe.twin.next

#             # Fix vertex assignments
#             targetv.halfedge = targetbhe[0]
#             newtargetv.halfedge = newh2

#         # Loop through incident halfedges of original boundary he and reassign vertex
#         he_bounds[0].vertex = newv
#         tmp_he = he_bounds[0].twin.next
#         while tmp_he != he_bounds[0]:
#             tmp_he.vertex = newv
#             tmp_he = tmp_he.twin.next
#         he.vertex = newv

class EdgeCut():
    def __init__(
        self, mesh, e_i, startv, splitf, cutbdry=False, e2_i = None
    ):
        """ e_i: index of edge to cut
            startv: where to start the cut (determines the direction of cut)
            splitf: index of face to assign the new primitives to (must be adjacent to the cut edge)
            e2_i: index of second edge to cut (must be specified if cutting middle of mesh)
            """
        self.mesh = mesh
        self.e_i = e_i
        self.cutbdry = cutbdry

        # Unit testing
        assert startv in [v.index for v in mesh.topology.edges[e_i].two_vertices()], f"Start vertex must be on the chosen cut edge!"

        self.startv = startv

        e_faces = [mesh.topology.edges[e_i].halfedge.face.index, mesh.topology.edges[e_i].halfedge.twin.face.index]
        assert splitf in e_faces, f"splitf needs to be one of the faces adjacent to the cut edge!"

        self.splitf = splitf

        if e2_i is not None:
            assert startv in [v.index for v in mesh.topology.edges[e2_i].two_vertices()], f"Start vertex must be on e2_i!"

        # if splitf2 is not None:
        #     e_faces = [mesh.topology.edges[e2_i].halfedge.face.index, mesh.topology.edges[e2_i].halfedge.twin.face.index]
        #     assert splitf2 in e_faces, f"splitf2 needs to be one of the faces adjacent to e2_i!"

        #     # Splitf2 and splitf must be on same side (share an edge)
        #     splitf_es = set(mesh.topology.faces[splitf].adjacentEdges())
        #     splitf2_es = set(mesh.topology.faces[splitf2].adjacentEdges())
        #     assert len(splitf_es.intersection(splitf2_es)) == 1, f"splitf and splitf2 must share an edge!"

        self.e2_i = e2_i

    def apply(self):
        he = self.mesh.topology.edges[self.e_i].halfedge

        # NOTE: halfedge should be on specified face
        if he.face.index != self.splitf:
            he = he.twin
            assert he.face.index == self.splitf, f"Neither halfedge on edge is incident to the chosen splitf face."

        sourcev = self.mesh.topology.vertices[self.startv]
        targetv = he.tip_vertex() if sourcev != he.tip_vertex() else he.vertex
        otherhe = he.twin

        #### Case 1: only 1 edge specified (startv must be on boundary)
        # Cut: 1 new vertex, 1 new edge, two new halfedges
        # New edge -> one new halfedge
        # New halfedges -> next each other, twin with cut faces, 1 old vertex, other next vertex, split faces, 1 new edge, 1 old edge
        # New vertex -> assign to hes up to cut edge
        if self.e2_i is None:
            assert sourcev.onBoundary(), f"If only one edge specified, then startv must be on boundary!"

            newv = self.mesh.topology.vertices.allocate()
            newe = self.mesh.topology.edges.allocate()
            newh1 = self.mesh.topology.halfedges.allocate()
            newh2 = self.mesh.topology.halfedges.allocate()

            # New vertex position
            self.mesh.vertices = np.concatenate([self.mesh.vertices, np.array([self.mesh.vertices[sourcev.index]])], axis=0)

            # Start the sweep from the boundary halfedge
            he_bounds = [halfe for halfe in sourcev.adjacentHalfedges() if halfe.onBoundary]
            assert len(he_bounds) == 1, f"Source vertex has {len(he_bounds)} boundaries."

            # Assign primitives
            # NOTE: If mesh consistently oriented, then he_bounds and he will be on opposite sides of the cut!!!
            # Splitf side (newv, newe, newh1 on this side)
            ## Subcase 1: vertex not incident on he
            if otherhe.vertex == sourcev:
                sourcev.halfedge = otherhe
                newv.halfedge = newh1
                newh1.vertex = newv

                newh1.onBoundary = True
                newh1.twin = he
                otherhe.twin = newh2
                he.twin = newh1
                newh1.edge = newe
                he.edge = newe
                newe.halfedge = newh1
                newh1.face = he_bounds[0].face
                newh1.next = newh2
                he_bounds[0].prev().next = newh1

                # Otherhe side (newh2 on this side)
                otherhe.edge.halfedge = otherhe
                newh2.edge = otherhe.edge

                assert otherhe.tip_vertex() == targetv
                newh2.vertex = targetv
                targetv.halfedge = newh2
                newh2.onBoundary = True
                newh2.next = he_bounds[0]
                newh2.twin = otherhe
                otherhe.twin = newh2
                newh2.face = he_bounds[0].face

                # Reassign vertex fans (on splitf side)
                currenthe = newh1
                while currenthe.twin.next != newh1:
                    currenthe = currenthe.twin.next
                    currenthe.vertex = newv
                assert newh1.vertex == newv

                ## Test topology
                self.mesh.topology.compactify_keys()
                self.mesh.topology.thorough_check()

                ## Check for targetv split condition (if targetv is also on a boundary)
                bd1 = he_bounds[0].face

                targetbhe = [halfe for halfe in targetv.adjacentHalfedges() if halfe.onBoundary and halfe != newh2 and halfe != newh1]
                assert len(targetbhe) <= 1, f"Can't handle targetv with more than two boundaries! ({len(targetbhe)})"
                if len(targetbhe) == 1 and self.cutbdry:
                    newtargetv = self.mesh.topology.vertices.allocate()
                    bdhe = targetbhe[0]

                    self.mesh.vertices = np.concatenate([self.mesh.vertices, np.array([self.mesh.vertices[targetv.index]])], axis=0)

                    ## Remove second boundary if different from original
                    if bdhe.face != bd1:
                        del self.mesh.topology.boundaries[bdhe.face.index]

                    ## Reassign boundary topology
                    bdhe_other = bdhe.prev()

                    targetv.halfedge = bdhe_other.twin
                    bdhe.face = newh1.face
                    newh1.next = bdhe
                    bdhe.vertex = newtargetv
                    newtargetv.halfedge = bdhe

                    bdhe_other.face = newh2.face
                    bdhe_other.next = newh2

                    ## Reassign vertex fans
                    currenthe = bdhe
                    while currenthe != he:
                        currenthe = currenthe.twin.next
                        currenthe.vertex = newtargetv
                    assert he.vertex == newtargetv

                    ## Loop through full boundary and reassign boundary index
                    for he in bd1.adjacentHalfedges():
                        if he.face != bd1:
                            he.face = bd1

                    ## Test topology
                    self.mesh.topology.compactify_keys()
                    self.mesh.topology.thorough_check()

                    assert len([v for v in bdhe.face.adjacentVertices()]) == len([v for v in self.mesh.topology.boundaries[newh1.face.index].adjacentVertices()]), f"Boundaries lengths not the same after targetv split!"

            ## Subcase 2: vertex incident on he
            else:
                newv.halfedge = he
                he.vertex = newv
                otherhe.vertex.halfedge = newh1
                newh1.vertex = otherhe.vertex
                newh1.next = he_bounds[0]
                newh1.onBoundary = True
                newh1.twin = he
                otherhe.twin = newh2
                he.twin = newh1
                newh1.edge = newe
                he.edge = newe
                newe.halfedge = newh1
                newh1.face = he_bounds[0].face

                # Otherhe side (newh2 on this side)
                he_bounds[0].prev().next = newh2
                otherhe.edge.halfedge = otherhe
                newh2.edge = otherhe.edge

                assert otherhe.tip_vertex() == sourcev
                newh2.vertex = sourcev
                sourcev.halfedge = newh2
                newh2.onBoundary = True
                newh2.next = newh1
                newh2.twin = otherhe
                otherhe.twin = newh2
                newh2.face = he_bounds[0].face

                # Reassign vertex fans (on splitf side)
                currenthe = he
                while currenthe.twin.next != he:
                    currenthe = currenthe.twin.next
                    currenthe.vertex = newv
                assert he.vertex == newv

                ## Test topology
                self.mesh.topology.compactify_keys()
                self.mesh.topology.thorough_check()

                ## Check for targetv split condition
                bd1 = he_bounds[0].face
                targetbhe = [halfe for halfe in targetv.adjacentHalfedges() if halfe.onBoundary and halfe != newh1 and halfe != newh2]
                assert len(targetbhe) <= 1, f"Can't handle targetv with more than two boundaries! ({len(targetbhe)})"

                # Targetv on same boundary as sourcev
                if len(targetbhe) == 1 and self.cutbdry:
                    newtargetv = self.mesh.topology.vertices.allocate()
                    bdhe = targetbhe[0]
                    self.mesh.vertices = np.concatenate([self.mesh.vertices, np.array([self.mesh.vertices[targetv.index]])], axis=0)

                    ## Remove second boundary if different from original
                    if bdhe.face != bd1:
                        del self.mesh.topology.boundaries[bdhe.face.index]

                    ## Reassign boundary topology
                    bdhe_other = bdhe.prev()
                    bdhe.face = newh2.face
                    newh2.next = bdhe
                    targetv.halfedge = bdhe
                    assert bdhe.vertex == targetv, f"Targetv boundary halfedge should be assigned to targetv!"

                    bdhe_other.face = newh1.face
                    bdhe_other.next = newh1
                    newtargetv.halfedge = newh1
                    newh1.vertex = newtargetv

                    ## Reassign vertex fans
                    currenthe = newh1
                    while currenthe != bdhe_other.twin:
                        currenthe = currenthe.twin.next
                        currenthe.vertex = newtargetv
                    assert bdhe_other.twin.vertex == newtargetv

                    ## Loop through full boundary and reassign boundary index
                    for he in bd1.adjacentHalfedges():
                        if he.face != bd1:
                            he.face = bd1

                    ## Test topology
                    self.mesh.topology.compactify_keys()
                    self.mesh.topology.thorough_check()

                    assert len([v for v in bdhe.face.adjacentVertices()]) == len([v for v in self.mesh.topology.boundaries[newh1.face.index].adjacentVertices()]), f"Boundaries lengths not the same after targetv split!"

            # Update faces
            vs, fs, es = self.mesh.export_soup()
            self.mesh.faces = fs

        #### Case 2: cut two edges in the middle of mesh
        # Cut: 1 new vertex, 2 new edges, 4 new halfedges, 1 new boundary (all new he on it)
        # New vertex -> assign to splitf1, splitf2
        # New edges -> 1 for ei, ei2
        # New halfedges -> all on boundary, twins of respective existing he
        else:
            newv = self.mesh.topology.vertices.allocate()

            newe1 = self.mesh.topology.edges.allocate()
            newe2 = self.mesh.topology.edges.allocate()

            newh1 = self.mesh.topology.halfedges.allocate()
            newh2 = self.mesh.topology.halfedges.allocate()
            newh3 = self.mesh.topology.halfedges.allocate()
            newh4 = self.mesh.topology.halfedges.allocate()

            newbd = self.mesh.topology.boundaries.allocate()

            # New vertex position
            self.mesh.vertices = np.concatenate([self.mesh.vertices, np.array([self.mesh.vertices[sourcev.index]])], axis=0)

            ### Subcase 1: sourcev is incident to he (first face)
            if he.vertex == sourcev:
                ## If sourcev on he, then splitf2 is on opposite side of fan
                currenthe = he
                while currenthe.edge.index != self.e2_i:
                    currenthe = currenthe.twin.next

                # Set primitives
                he2 = currenthe.twin
                splitf2 = he2.face
                otherhe2 = he2.twin
                targetv2 = he2.vertex

                ## Set boundary
                newh1.onBoundary = True
                newh2.onBoundary = True
                newh3.onBoundary = True
                newh4.onBoundary = True

                newh1.face = newbd
                newh2.face = newbd
                newh3.face = newbd
                newh4.face = newbd

                newh1.next = newh2
                newh2.next = newh3
                newh3.next = newh4
                newh4.next = newh1

                newh1.twin = he
                newh2.twin = he2
                newh3.twin = otherhe2
                newh4.twin = otherhe

                he.twin = newh1
                he2.twin = newh2
                otherhe2.twin = newh3
                otherhe.twin = newh4

                newbd.halfedge = newh1

                ## Splitf, splitf2 side (newe1, newe2, newh1, newh4, newv all on this side)
                newv.halfedge = he
                he.vertex = newv
                he.edge = newe1
                newh1.vertex = he.tip_vertex()
                he.tip_vertex().halfedge = newh1
                newh1.edge = newe1
                newe1.halfedge = he

                newh2.vertex = newv
                newh2.edge = newe2
                he2.edge = newe2
                newe2.halfedge = he2

                ## Otherhe side (newh2, newh3 all on this side)
                targetv2.halfedge = newh3
                newh3.vertex = targetv2
                newh3.edge = otherhe2.edge
                otherhe2.edge.halfedge = newh3

                newh4.vertex = sourcev
                sourcev.halfedge = newh4
                newh4.edge = otherhe.edge
                otherhe.edge.halfedge = newh4

                # Reassign vertex fans (on splitf/splitf2 side)
                currenthe = newh2
                stophe = he
                while currenthe != stophe:
                    currenthe = currenthe.twin.next
                    currenthe.vertex = newv
                assert stophe.vertex == newv

                ## Test topology
                self.mesh.topology.compactify_keys()
                self.mesh.topology.thorough_check()

                ## Check for targetv2 split condition
                bd1 = newbd
                targetbhe = [halfe for halfe in targetv2.adjacentHalfedges() if halfe.onBoundary and halfe != newh3]
                assert len(targetbhe) <= 1, f"Can't handle targetv with more than two boundaries! ({len(targetbhe)})"

                if len(targetbhe) == 1 and self.cutbdry:
                    newtargetv = self.mesh.topology.vertices.allocate()
                    bdhe = targetbhe[0]
                    self.mesh.vertices = np.concatenate([self.mesh.vertices, np.array([self.mesh.vertices[targetv2.index]])], axis=0)

                    ## Remove old boundary
                    if bdhe.face != bd1:
                        del self.mesh.topology.boundaries[bdhe.face.index]

                    ## Reassign boundary topology
                    bdhe_other = bdhe.prev()

                    targetv2.halfedge = bdhe_other.twin
                    bdhe.face = newh2.face
                    newh2.next = bdhe
                    assert bdhe.vertex == targetv2, f"Second boundary he vertex should be targetv!"

                    bdhe_other.face = newh3.face
                    bdhe_other.next = newh3
                    newtargetv.halfedge = bdhe
                    bdhe.vertex = newtargetv

                    ## Reassign vertex fans
                    currenthe = bdhe
                    while currenthe != he2:
                        currenthe = currenthe.twin.next
                        currenthe.vertex = newtargetv
                    assert he2.vertex == newtargetv

                    ## Reassign boundary
                    for halfe in bd1.adjacentHalfedges():
                        halfe.face = bd1

                    ## Test topology
                    self.mesh.topology.compactify_keys()
                    self.mesh.topology.thorough_check()

                    assert len([v for v in bdhe.face.adjacentVertices()]) == len([v for v in self.mesh.topology.boundaries[newh1.face.index].adjacentVertices()]), f"Boundaries lengths not the same after targetv split!"

            ### Subcase 2: sourcev is tip of he
            else:
                ## If sourcev tip of he, then splitf2 is on same side as fan
                currenthe = he.next
                while currenthe.edge.index != self.e2_i:
                    currenthe = currenthe.twin.next

                # Set primitives
                he2 = currenthe
                splitf2 = he2.face
                otherhe2 = he2.twin
                targetv2 = otherhe2.vertex

                ## Set boundary
                newh1.onBoundary = True
                newh2.onBoundary = True
                newh3.onBoundary = True
                newh4.onBoundary = True

                newh1.face = newbd
                newh2.face = newbd
                newh3.face = newbd
                newh4.face = newbd

                newh1.next = newh4
                newh4.next = newh3
                newh3.next = newh2
                newh2.next = newh1

                newh1.twin = he
                newh2.twin = he2
                newh3.twin = otherhe2
                newh4.twin = otherhe

                he.twin = newh1
                he2.twin = newh2
                otherhe2.twin = newh3
                otherhe.twin = newh4

                newbd.halfedge = newh1

                ## Splitf, splitf2 side (newe1, newe2, newh1, newh2, newv all on this side)
                he2.vertex = newv
                he2.edge = newe2
                newh2.edge = newe2
                newh2.vertex = targetv2
                newe2.halfedge = he2
                targetv2.halfedge = newh2

                newv.halfedge = he2
                he.edge = newe1

                newh1.edge = newe1
                newe1.halfedge = he
                newh1.vertex = newv

                ## Otherhe side (newh2, newh3 all on this side)
                newh4.vertex = otherhe.tip_vertex()
                otherhe.tip_vertex().halfedge = newh4
                newh4.edge = otherhe.edge
                otherhe.edge.halfedge = newh4

                newh3.vertex = sourcev
                sourcev.halfedge = newh3
                newh3.edge = otherhe2.edge
                otherhe2.edge.halfedge = newh3

                # Reassign vertex fans (on splitf/splitf2 side)
                currenthe = newh1
                stophe = he2
                while currenthe != stophe:
                    currenthe = currenthe.twin.next
                    currenthe.vertex = newv
                assert stophe.vertex == newv

                ## Test topology
                self.mesh.topology.compactify_keys()
                self.mesh.topology.thorough_check()

                ## Check for targetv2 split condition
                targetbhe = [halfe for halfe in targetv2.adjacentHalfedges() if halfe.onBoundary and halfe != newh2]
                assert len(targetbhe) <= 1, f"Can't handle targetv with more than two boundaries! ({len(targetbhe)})"

                if len(targetbhe) == 1 and self.cutbdry:
                    newtargetv = self.mesh.topology.vertices.allocate()
                    bdhe = targetbhe[0]
                    self.mesh.vertices = np.concatenate([self.mesh.vertices, np.array([self.mesh.vertices[targetv.index]])], axis=0)

                    ## Remove old boundary
                    if bdhe.face != newbd:
                        del self.mesh.topology.boundaries[bdhe.face.index]

                    ## Reassign boundary topology
                    bdhe_other = bdhe.prev()
                    bdhe.face = newh4.face
                    newh3.next = bdhe
                    newh2.vertex = newtargetv
                    newtargetv.halfedge = newh2

                    # In case targetv2 is assigned to split side
                    targetv2.halfedge = bdhe

                    bdhe_other.face = newh2.face
                    bdhe_other.next = newh2

                    ## Reassign vertex fans
                    currenthe = newh2
                    while currenthe != bdhe_other.twin:
                        currenthe = currenthe.twin.next
                        currenthe.vertex = newtargetv
                    assert newh2.vertex == newtargetv

                    ## Reassign boundary
                    for halfe in newbd.adjacentHalfedges():
                        halfe.face = newbd

                    ## Test topology
                    self.mesh.topology.compactify_keys()
                    self.mesh.topology.thorough_check()

                    assert len([v for v in bdhe.face.adjacentVertices()]) == len([v for v in self.mesh.topology.boundaries[newh1.face.index].adjacentVertices()]), f"Boundaries lengths not the same after targetv split!"

            # Update faces
            vs, fs, es = self.mesh.export_soup()
            self.mesh.faces = fs

class VertexSplit():
    def __init__(
        self, mesh, v_top_id, e_left_id, e_right_id, v_top_coord, v_bottom_coord,
        v_bottom_id=None, new_e_bundle = None, new_e_left_bundle = None, new_e_right_bundle = None,
        new_f_bundle=None, makebdry=False
    ):
        # ignore the offset; do the topology editing properly first
        topology = mesh.topology
        v = topology.vertices[v_top_id]
        e1 = topology.edges[e_left_id]
        e2 = topology.edges[e_right_id]
        assert v in e1.two_vertices()
        assert v in e2.two_vertices()

        self.mesh = mesh

        self.v_top_id = v_top_id
        self.v_bottom_id = v_bottom_id
        self.e_left_id = e_left_id
        self.e_right_id = e_right_id

        self.v_top_coord = v_top_coord
        self.v_bottom_coord = v_bottom_coord

        self.new_e_bundle = new_e_bundle
        self.new_e_left_bundle = new_e_left_bundle
        self.new_e_right_bundle = new_e_right_bundle
        self.new_f_bundle = new_f_bundle

        # Makes a boundary loop instead of a new edge connecting the split vertex
        self.makebdry = makebdry

    def apply(self):
        topology = self.mesh.topology
        v = topology.vertices[self.v_top_id]
        e1 = topology.edges[self.e_left_id]
        e2 = topology.edges[self.e_right_id]

        heA1 = e1.halfedge if e1.halfedge.vertex == v else e1.halfedge.twin
        heB2 = e2.halfedge if e2.halfedge.tip_vertex() == v else e2.halfedge.twin

        vB = heA1.vertex
        assert vB == v
        self.mesh.vertices[vB.index] = self.v_top_coord
        # assert self.v_bottom_id is not None
        if self.v_bottom_id is None:
            vA = topology.vertices.allocate()
            self.mesh.vertices = np.append(
                self.mesh.vertices,
                self.v_bottom_coord.reshape(1, -1),
                axis=0
            )
            self.v_bottom_id = vA.index
            assert vA.index == (len(self.mesh.vertices) - 1)
        else:
            vA = topology.vertices.fill_vacant(self.v_bottom_id)
            self.mesh.vertices[vA.index] = self.v_bottom_coord

        def allocate_edge_bundle(e_id, he1_id, he2_id):
            if e_id is not None and he1_id is not None and he2_id is not None:
                e = topology.edges.fill_vacant(e_id)
                he1 = topology.halfedges.fill_vacant(he1_id)
                he2 = topology.halfedges.fill_vacant(he2_id)
            else:
                e = topology.edges.allocate()
                he1 = topology.halfedges.allocate()
                he2 = topology.halfedges.allocate()
            he1.twin = he2
            he2.twin = he1
            he1.edge = e
            he2.edge = e
            e.halfedge = he1
            # don't even have to return the e; just stitch the halfedge in
            return he1, he2

        v_top_id, v_bottom_id, e_left_id, e_right_id, v_top_coord, v_bottom_coord, new_e_bundle, new_e_left_bundle, new_e_right_bundle = self.record
        record = (
            vB.index, vA.index, heA1.edge.index, heB2.edge.index,
            mesh.vertices[vB.index].copy(), mesh.vertices[vA.index].copy(),
            (e_id, e.halfedge.index, e.halfedge.twin.index),
            (heA2.edge.index, heA2.index, heA2.twin.index),
            (heB1.edge.index, heB1.index, heB1.twin.index)
        )

        if heB2.next == heA1:
            # this is the valence 3 split case:
            heC0 = heA1.next
            fC = heC0.face

            vC = heC0.vertex
            vD = heB2.vertex

            heA2, heC2 = allocate_edge_bundle(self.new_e_left_bundle[0], self.new_e_left_bundle[1], self.new_e_left_bundle[2])
            heA0, heB0 = allocate_edge_bundle(self.new_e_bundle[0], self.new_e_bundle[1], self.new_e_bundle[2])
            heB1, heC1 = allocate_edge_bundle(self.new_e_right_bundle[0], self.new_e_right_bundle[1], self.new_e_right_bundle[2])

            if self.new_f_bundle is not None:
                fA = topology.faces.fill_vacant(self.new_f_bundle[0])
                fB = topology.faces.fill_vacant(self.new_f_bundle[1])
            else:
                fA = topology.faces.allocate()
                fB = topology.faces.allocate()

            # Set boundary conditions
            if heA1.onBoundary == True:
                heA1.onBoundary = False
                heB2.onBoundary = False
                heC1.onBoundary = True
                heC2.onBoundary = True
                heB2.prev().next = heC1
                if self.mesh.topology.boundaries[0].halfedge in [heA1, heB2]:
                    self.mesh.topology.boundaries[0].halfedge = heC1
            # he's next
            heA1.next = heA2
            heA2.next = heA0
            heA0.next = heA1

            heB0.next = heB1
            heB1.next = heB2
            heB2.next = heB0

            heC1.next = heC2
            heC2.next = heC0
            if heC0.onBoundary == False:
                heC0.next = heC1

            # he's twin; no need

            # he's v
            heA0.vertex = vA
            heA1.vertex = vB
            heA2.vertex = vC

            heC0.vertex = vC
            heC1.vertex = vD
            heC2.vertex = vA

            heB0.vertex = vB
            heB1.vertex = vA
            heB2.vertex = vD

            # he's f
            heA0.face = fA
            heA1.face = fA
            heA2.face = fA
            heB0.face = fB
            heB1.face = fB
            heB2.face = fB
            # It's okay if fC is boundary face here
            heC0.face = fC
            heC1.face = fC
            heC2.face = fC

            # he's e
            # v's he
            vA.halfedge = heA0
            vB.halfedge = heA1
            vC.halfedge = heC0
            vD.halfedge = heB2

            # f's he
            fA.halfedge = heA0
            fB.halfedge = heB0
            fC.halfedge = heC0
            # e's he
        else:
            # this is the regular valence split case
            # TODO: boundary cases are special here -- only 1 edge on boundary!
            heC0 = heA1.next
            heC1 = heC0.next

            heD2 = heB2.next
            heD0 = heD2.next

            fC = heC0.face
            fD = heD0.face

            vC = heC0.vertex
            vD = heB2.vertex

            heA2, heC2 = allocate_edge_bundle(self.new_e_left_bundle[0], self.new_e_left_bundle[1],
                                              self.new_e_left_bundle[2])
            heA0, heB0 = allocate_edge_bundle(self.new_e_bundle[0], self.new_e_bundle[1], self.new_e_bundle[2])
            heB1, heD1 = allocate_edge_bundle(self.new_e_right_bundle[0], self.new_e_right_bundle[1],
                                              self.new_e_right_bundle[2])

            if self.new_f_bundle is not None:
                fA = topology.faces.fill_vacant(self.new_f_bundle[0])
                fB = topology.faces.fill_vacant(self.new_f_bundle[1])
            else:
                fA = topology.faces.allocate()
                fB = topology.faces.allocate()

            # he's next
            heA0.next = heA1
            heA1.next = heA2
            heA2.next = heA0
            heB0.next = heB1
            heB1.next = heB2
            heB2.next = heB0
            heC0.next = heC1
            heC1.next = heC2
            heC2.next = heC0
            heD0.next = heD1
            heD1.next = heD2
            heD2.next = heD0

            # he's twin; no need
            # he's v
            heA0.vertex = vA
            heA1.vertex = vB
            heA2.vertex = vC
            heB0.vertex = vB
            heB1.vertex = vA
            heB2.vertex = vD
            heC0.vertex = vC
            heC2.vertex = vA
            heD1.vertex = vD
            heD2.vertex = vA

            # while loop
            he_itr = heC2
            while True:
                he_itr.vertex = vA
                he_itr = he_itr.twin.next
                if he_itr == heC2:
                    break
            del he_itr

            # he's f
            heA0.face = fA
            heA1.face = fA
            heA2.face = fA
            heB0.face = fB
            heB1.face = fB
            heB2.face = fB
            heC0.face = fC
            heC1.face = fC
            heC2.face = fC
            heD0.face = fD
            heD1.face = fD
            heD2.face = fD

            # he's e
            # v's he
            vA.halfedge = heA0
            vB.halfedge = heA1
            vC.halfedge = heC0
            vD.halfedge = heB2

            # f's he
            fA.halfedge = heA0
            fB.halfedge = heB0
            fC.halfedge = heC0
            fD.halfedge = heD0
            # e's he

class VertexStarCollapse(MeshEdit):
    def __init__(self, mesh, v_id):
        self.mesh = mesh
        self.v_id = v_id
        self.original_topology = None

    def apply(self, no_degen=False, debug=False):
        # Save original topology
        self.original_topology = self.mesh.topology.export_halfedge_serialization()

        # Algorithm:
        # -- To delete: all one-ring half-edges, all boundary half-edges, and all adjacent face half-edges
        # -- Generate to delete half edge indices + full edge indices (everything in 1-ring) + 1-ring faces + boundary faces
        # -- For each boundary vertex, update each half-edge vertex to be star vertex
        # -- For non-boundary face: update the twin half-edge for the edge pointing towards the boundary vertex
        #    to the next valid face half-edge (will have to loop with stopping condition + error handling)
        # -- TODO: edge case if vertex on mesh boundary
        vertex = self.mesh.topology.vertices[self.v_id]
        onering_he = vertex.adjacentHalfedges()
        onering_he = [he for he in onering_he]

        # (1) Compile indices to delete
        boundary_he = [he.next.twin for he in onering_he]
        delete_he = [he.index for he in onering_he] + [he.twin.index for he in onering_he]
        delete_f = [he.face.index for he in onering_he]
        delete_e = [he.edge.index for he in onering_he]
        delete_v = []
        delete_boundary = []
        i = 0
        while i < len(boundary_he):
            he = boundary_he[i]
            delete_f.append(he.face.index)
            delete_e.append(he.edge.index)
            delete_v.append(he.vertex.index)
            delete_he.append(he.index)
            delete_he.append(he.twin.index)
            delete_he.append(he.next.index)
            delete_he.append(he.next.next.index)

            # Check for edge case: next boundary edge shares face
            next_he = boundary_he[(i + len(boundary_he) + 1) % len(boundary_he)]
            j = i
            next_ind = (j + len(boundary_he) + 1) % len(boundary_he)
            while next_he.face == he.face:
                delete_boundary.append(next_ind)
                # Update boundary
                he = he.next.next.twin
                delete_f.append(he.face.index)
                delete_e.append(he.edge.index)
                delete_v.append(he.vertex.index)
                delete_he.append(he.index)
                delete_he.append(he.twin.index)
                delete_he.append(he.next.index)
                delete_he.append(he.next.next.index)
                boundary_he[i] = he
                j += 1
                next_ind = (j + len(boundary_he) + 1) % len(boundary_he)
                next_he = boundary_he[next_ind]
            i += 1
        # (1.5) Delete from boundary if necessary
        import numpy as np
        boundary_he = np.array(boundary_he)
        if len(delete_boundary) > 0:
            keep = list(set(range(len(boundary_he))).difference(set(delete_boundary)))
            boundary_he = boundary_he[keep]

        # (2) Update the reference vertex for all the half-edges on boundary
        for he in boundary_he:
            for tmp_he in he.vertex.adjacentHalfedges():
                # if debug == True:
                #     import polyscope as ps
                #     import numpy as np
                #     ps.init()
                #     new_v, new_f, _ = self.mesh.export_soup()
                #     new_mesh = ps.register_surface_mesh("mesh", new_v, new_f, edge_width=1)
                #     old_inds = np.array(sorted(self.mesh.topology.vertices.keys()))
                #     v_colors = np.zeros(len(self.mesh.vertices))
                #     v_colors[tmp_he.vertex.index] = 1
                #     v_colors[tmp_he.next.vertex.index] = 2
                #     v_colors = v_colors[old_inds]
                #     new_mesh.add_scalar_quantity("half edge", v_colors, defined_on="vertices", enabled=True)
                #     ps.show()
                #     v_colors = np.zeros(len(self.mesh.vertices))
                #     v_colors[tmp_he.twin.vertex.index] = 1
                #     v_colors[tmp_he.twin.next.vertex.index] = 2
                #     v_colors = v_colors[old_inds]
                #     new_mesh.add_scalar_quantity("half edge", v_colors, defined_on="vertices", enabled=True)
                #     ps.show()
                if tmp_he.index not in delete_he:
                    tmp_he.vertex = vertex
                    vertex.halfedge = tmp_he

        # (3) Update twins
        for he in boundary_he:
            valid_he = he.next.next.twin
            # Edge case: adjacent face is also boundary face
            if valid_he.index in delete_he:
                # Add edge to delete
                delete_e.append(valid_he.edge.index)
                # If vertex half-edge is on the bad edge, then need to reset
                adj_v = valid_he.twin.vertex
                while adj_v.halfedge.index in delete_he:
                    adj_v.halfedge = adj_v.halfedge.twin.next
                continue
            candidate_he = valid_he.twin.next.next.twin
            while candidate_he.index in delete_he:
                candidate_he = candidate_he.next.next.twin
                # Stopping condition: full rotation around boundary
                if candidate_he.next.index == he.index:
                    print(f"Error: no valid twin found for boundary-adjacent half-edge {valid_he}")
            # Set twins
            valid_he.twin = candidate_he
            candidate_he.twin = valid_he
            # Vertex halfedge
            candidate_he.vertex.halfedge = candidate_he
            # Set edges
            valid_he.edge.halfedge = valid_he
            delete_e.append(candidate_he.edge.index)
            candidate_he.edge = valid_he.edge
            # TODO: Set vertices (???)
            # if debug == True:
            #     import polyscope as ps
            #     import numpy as np
            #     ps.init()
            #     new_v, new_f, _ = self.mesh.export_soup()
            #     new_mesh = ps.register_surface_mesh("mesh", new_v, new_f, edge_width=1)
            #     old_inds = np.array(sorted(self.mesh.topology.vertices.keys()))
            #     v_colors = np.zeros(len(self.mesh.vertices))
            #     v_colors[valid_he.vertex.index] = 1
            #     v_colors[valid_he.next.vertex.index] = 2
            #     v_colors = v_colors[old_inds]
            #     new_mesh.add_scalar_quantity("half edge", v_colors, defined_on="vertices", enabled=True)
            #     ps.show()
            #     v_colors = np.zeros(len(self.mesh.vertices))
            #     v_colors[valid_he.twin.vertex.index] = 1
            #     v_colors[valid_he.twin.next.vertex.index] = 2
            #     v_colors = v_colors[old_inds]
            #     new_mesh.add_scalar_quantity("half edge twin", v_colors, defined_on="vertices", enabled=True)
            #     ps.show()

        # (4) Update all vertices with halfedges assigned to deleted half-edges
        # for he_ind in delete_he:
        #     tmp_vertex = self.mesh.topology.halfedges[he_ind].vertex
        #     while tmp_vertex.index not in delete_v and tmp_vertex.halfedge.index in delete_he:
        #         tmp_vertex.halfedge = tmp_vertex.halfedge.twin.next
        #         if tmp_vertex.halfedge.index == he_ind:
        #             print(f"Error: no valid non-deleted halfedge to assign to non-deleted vertex.")
        #             import polyscope as ps
        #             import numpy as np
        #             ps.init()
        #             new_v, new_f, _ = self.mesh.export_soup()
        #             new_mesh = ps.register_surface_mesh("mesh", new_v, new_f, edge_width=1)
        #             for he in tmp_vertex.adjacentHalfedges():
        #                 old_inds = np.array(sorted(self.mesh.topology.vertices.keys()))
        #                 v_colors = np.zeros(len(self.mesh.vertices))
        #                 v_colors[he.vertex.index] = 1
        #                 v_colors[he.next.vertex.index] = 2
        #                 v_colors = v_colors[old_inds]
        #                 new_mesh.add_scalar_quantity("valid vertex", v_colors, defined_on="vertices", enabled=True)
        #                 ps.show()
        #             # # Loop through deleted vertices
        #             # for ind in delete_v:
        #             #     if ind not in old_inds:
        #             #         print(f"Vertex {ind} no longer in mesh anymore")
        #             #         raise
        #             #     v_colors = np.zeros(len(self.mesh.vertices))
        #             #     v_colors[ind] = 1
        #             #     v_colors = v_colors[old_inds]
        #             #     new_mesh.add_scalar_quantity("deleted vertex", v_colors, defined_on="vertices", enabled=True)
        #             #     ps.show()
        #             break

        # Checks: orphaned half-edges and vertices
        for v_ind in delete_v:
            v = self.mesh.topology.vertices[v_ind]
            he = v.halfedge
            if he.vertex.index in delete_v and he.index not in delete_he:
                print(f"Orphaned halfedge {he} found")
        for he_ind in delete_he:
            he = self.mesh.topology.halfedges[he_ind]
            if he.vertex.index not in delete_v and he.vertex.halfedge.index in delete_he:
                print(f"Orphaned vertex {he.vertex} found")

        # NOTE: Duplicates will arise when adjacent boundary edges share same face
        delete_f = np.unique(delete_f)
        delete_e = np.unique(delete_e)
        delete_v = np.unique(delete_v)
        delete_he = np.unique(delete_he)
        # Delete everything that was scheduled
        for he_ind in delete_he:
            del self.mesh.topology.halfedges[he_ind]
        for f_ind in delete_f:
            del self.mesh.topology.faces[f_ind]
        for e_ind in delete_e:
            del self.mesh.topology.edges[e_ind]
        for v_ind in delete_v:
            del self.mesh.topology.vertices[v_ind]

        # Check one-star
        if debug == True:
            import polyscope as ps
            import numpy as np

            for he in vertex.adjacentHalfedges():
                ps.init()
                new_v, new_f, _ = self.mesh.export_soup()
                new_mesh = ps.register_surface_mesh("mesh", new_v, new_f, edge_width=1)
                old_inds = np.array(sorted(self.mesh.topology.vertices.keys()))
                v_colors = np.zeros(len(self.mesh.vertices))
                v_colors[he.vertex.index] = 1
                v_colors[he.next.vertex.index] = 2
                v_colors = v_colors[old_inds]
                new_mesh.add_scalar_quantity("half edge", v_colors, defined_on="vertices", enabled=True)
                ps.show()
                v_colors = np.zeros(len(self.mesh.vertices))
                v_colors[he.twin.vertex.index] = 1
                v_colors[he.twin.next.vertex.index] = 2
                v_colors = v_colors[old_inds]
                new_mesh.add_scalar_quantity("half edge twin", v_colors, defined_on="vertices", enabled=True)
                ps.show()

        return vertex

    def inverse(self):
        # Recover the original topology
        if self.original_topology is not None:
            return self.original_topology
        return None