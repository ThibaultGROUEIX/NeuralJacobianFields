class UnionFind():
    def __init__(self, n):
        self.orig_n = n
        self.n_components = n
        self.leaders = list(range(n))

    def __str__(self) -> str:
        return str(self.leaders)

    def __repr__(self) -> str:
        return str(self)

    def merge_into(self, trg: int, src: int):
        """this op is ordered: merge src (source) into trg (target)"""
        a_lead = self.find(trg)
        b_lead = self.find(src)
        if a_lead == b_lead:
            return
        self.n_components -= 1
        new_lead = a_lead
        self.leaders[a_lead] = new_lead
        self.leaders[b_lead] = new_lead

    def find(self, inx: int):
        lead = self.leaders[inx]
        lead_lead = self.leaders[lead]
        if lead_lead == lead:
            return lead
        else:
            self.leaders[inx] = self.find(lead_lead)
            return self.leaders[inx]

    def _find(self, inx: int):
        # this is slightly slower
        paths = []
        lead = self.leaders[inx]
        while (self.leaders[lead] != lead):
            paths.append(inx)
            inx = lead
            lead = self.leaders[inx]
        true_lead = lead
        for node in paths:
            self.leaders[node] = true_lead
        return true_lead

    """these two methods below are currently not used
    Since MeshCNN pools the average of the 3 edges in each halfflap,
    uf is not suitable for tracking such changes.
    """
    def to_sparse_transition_matrix(self):
        import torch
        m, n = self.n_components, self.orig_n
        inds = []
        mapping = self.group_mapping()
        for j in range(n):
            i = self.find(j)
            inds.append([mapping[i], j])
        inds = list(zip(*inds))
        vals = [1, ] * n
        trans = torch.sparse_coo_tensor(inds, vals, (m, n), dtype=torch.float)
        return trans

    def group_mapping(self):
        # first consolidate all leaders
        for i in range(self.orig_n):
            self.find(i)

        mapping = {}
        grp_inx = 0
        for i in sorted(self.leaders):  # this sorting is critical
            if i not in mapping:
                mapping[i] = grp_inx
                grp_inx += 1
        return mapping


def test():
    uf = UnionFind(10)
    uf.union(2, 4)
    uf.union(0, 2)
    print(uf)
    print(uf.find(3))
    print(uf.find(4))
    print(uf)


if __name__ == '__main__':
    test()
