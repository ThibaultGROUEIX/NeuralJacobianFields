class Primitive():
    def __init__(self):
        self.halfedge = None
        self.index = -1

        # for quadric error simplification
        self.quadrics = None

    def __str__(self) -> str:
        return str(self.index)

    def __repr__(self) -> str:
        return str(self)


class Halfedge(Primitive):
    def __init__(self):
        # note parent constructor is replaced
        self.vertex = None
        self.edge = None
        self.face = None
        # self.corner = None
        self.next = None
        self.twin = None
        self.onBoundary = False
        self.index = -1  # an ID between 0 and |H| - 1, where |H| is the number of halfedges in a mesh

    def prev(self):
        # this is constant cost on trimesh; maintaining the pointers is error prone
        he = self
        while he.next != self:
            he = he.next
        return he

    def tip_vertex(self):
        return self.next.vertex

    def serialize(self):
        return (
            self.index,
            self.vertex.index,
            self.edge.index,
            self.face.index,
            self.next.index,
            self.twin.index,
            int(self.onBoundary)
        )

        i_he, i_vert, i_edge, i_face, i_next, i_twin, is_bound

    # def onBoundary(self):
    #     return (self.vertex.onBoundary() or self.tip_vertex().onBoundary())

class Edge(Primitive):
    def onBoundary(self):
        return (self.halfedge.onBoundary or self.halfedge.twin.onBoundary)

    @property
    def id(self):
        id_str = self.generate_eid(
            self.halfedge.vertex.index, self.halfedge.tip_vertex().index
        )
        return id_str

    @classmethod
    def generate_eid(cls, vid_1, vid_2):
        vid_a, vid_b = (vid_1, vid_2) if vid_1 < vid_2 else (vid_2, vid_1)
        id_str = f"{vid_a}-{vid_b}"
        return id_str

    def two_vertices(self):
        """return the two incident vertices of the edge
        note that the incident vertices are ambiguous to ordering
        """
        return (self.halfedge.vertex, self.halfedge.tip_vertex())

class Face(Primitive):
    def isBoundaryLoop(self):
        return self.halfedge.onBoundary

    def adjacentHalfedges(self):
        current = self.halfedge
        end = self.halfedge
        # halfedges = [current, current.next, current.next.next]
        # return halfedges
        while True:
            he = current
            yield he
            current = current.next
            if current == end:
                break

    def adjacentVertices(self):
        for he in self.adjacentHalfedges():
            yield he.vertex
        # return [he.vertex for he in self.adjacentHalfedges()]

    def adjacentEdges(self):
        # return [he.edge for he in self.adjacentHalfedges()]
        for he in self.adjacentHalfedges():
            yield he.edge

    def adjacentFaces(self):
        # return [he.twin.face for he in self.adjacentHalfedges() if not he.twin.face.isBoundaryLoop()]
        for he in self.adjacentHalfedges():
            face = he.twin.face
            if face.isBoundaryLoop():
                pass
            else:
                yield face


class Vertex(Primitive):
    def degree(self):
        k = 0
        for _ in self.adjacentEdges():
            k += 1
        return k

    @classmethod
    def merge_into(cls, mesh, vA, vB):
        """merge vA into vB; for now do simple midpoint merge"""
        mesh.vertices[vB.index] = (mesh.vertices[vA.index] + mesh.vertices[vB.index]) / 2
        if vA.quadrics is not None and vB.quadrics is not None:
            vB.quadrics = vA.quadrics + vB.quadrics

    def isIsolated(self) -> bool:
        return self.halfedge is None

    def onBoundary(self):
        for he in self.adjacentHalfedges():
            if he.onBoundary:
                return True
        return False

    def adjacentHalfedges(self):
        current = self.halfedge
        end = self.halfedge
        # halfedges = []
        while True:
            he = current
            # halfedges.append(he)
            # print(he, end='\r')
            yield he
            current = current.twin.next
            if current == end:
                break
        # return halfedges

    def adjacentVertices(self):
        # return [he.twin.vertex for he in self.adjacentHalfedges()]
        for he in self.adjacentHalfedges():
            yield he.twin.vertex

    def adjacentEdges(self):
        # return [he.edge for he in self.adjacentHalfedges()]
        for he in self.adjacentHalfedges():
            yield he.edge

    def adjacentFaces(self):
        # return [he.face for he in self.adjacentHalfedges() if not he.face.isBoundaryLoop()]
        for he in self.adjacentHalfedges():
            face = he.face
            if face.isBoundaryLoop():
                pass
            else:
                yield face
