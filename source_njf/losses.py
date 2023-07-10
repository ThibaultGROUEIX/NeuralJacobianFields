# InteractiveSegmentation: Standardized set of tests for parameterization loss functions
import numpy as np
import torch

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

def meshArea2D(vertices, faces, return_fareas = False):
    # Get edge vectors
    fverts = vertices[faces]
    edge1 = fverts[:,1,:] - fverts[:,0,:]
    edge2 = fverts[:,2,:] - fverts[:,0,:]

    # Determinant definition of area
    area = 0.5 * torch.abs(edge1[:,0] * edge2[:,1]  - edge1[:,1] * edge2[:,0])

    # Debugging
    # print(area[0])
    # print(fverts[0])
    # exit()

    if return_fareas == True:
        return area
    else:
        return torch.sum(area)

# ==================== Loss Wrapper Class ===============================
### Naming conventions
### - "loss": will be plotted in the plot_uv function
### - "edge": loss will be plotted over edges, else triangles
from collections import defaultdict

class UVLoss:
    def __init__(self, args, device=torch.device("cpu")):
        self.args = args
        self.currentloss = defaultdict(dict)
        self.device = device
        self.count = 0

        ### Sanity checks
        # Can only have one of seploss or gradloss
        assert not (self.args.lossedgeseparation and self.args.lossgradientstitching), f"Can only have one type of edge loss!"

        # Record loss names (for visualization)
        self.lossnames = []
        if self.args.lossgt:
            self.lossnames.append("gtloss")

        if self.args.lossdistortion:
            self.lossnames.append("distortionloss")

        if self.args.losscount and self.args.lossdistortion:
            self.lossnames.append("countloss")

        if self.args.lossedgeseparation:
            self.lossnames.append("edgeseploss")

        if self.args.lossgradientstitching:
            self.lossnames.append("edgegradloss")

    def clear(self):
        self.currentloss = defaultdict(dict)
        self.count = 0

    def computeloss(self, vertices = None, faces = None, uv = None, jacobians = None, initjacobs=None, seplossdelta=0.1,
                    transuv=None, gtjacobians=None):
        loss = 0
        # Ground truth
        if self.args.lossgt and gtjacobians is not None:
            gtloss = torch.nn.functional.mse_loss(jacobians, gtjacobians, reduction='none')
            loss += torch.mean(gtloss)
            self.currentloss[self.count]['gtloss'] = torch.mean(gtloss, dim=[1,2]).detach().cpu().numpy()

        # Autocuts
        if self.args.lossautocut:
            acloss = autocuts(vertices, faces, jacobians, uv, self.args.seplossweight, seplossdelta)
            loss += acloss
            self.currentloss[self.count]['autocuts'] = acloss.detach().item()

        # Distortion
        distortionenergy = None
        if self.args.lossdistortion == "arap":
            # TODO: weight by 3D face areas?
            local_tris = get_local_tris(vertices, faces, device=self.device)
            distortionenergy = arap(local_tris, faces, paramtris=uv,
                                device=self.device, renormalize=False,
                                return_face_energy=True, timeit=False)
            self.currentloss[self.count]['distortionloss'] = distortionenergy.detach().cpu().numpy()

            if not self.args.losscount:
                loss += torch.mean(distortionenergy)

        if self.args.lossdistortion == "dirichlet":
            distortionenergy = symmetricdirichlet(vertices, faces, jacobians, init_jacob=initjacobs)
            self.currentloss[self.count]['distortionloss'] = distortionenergy.detach().cpu().numpy()

            if not self.args.losscount:
                loss += torch.mean(distortionenergy)

        if self.args.lossdistortion == "edge":
            distortionenergy = edgedistortion(vertices, faces, uv)
            self.currentloss[self.count]['distortionloss'] = distortionenergy.detach().cpu().numpy()

            if not self.args.losscount:
                loss += torch.mean(distortionenergy)

        if self.args.lossedgeseparation:
            edgeseploss, edgecorrespondences = uvseparation(vertices, faces, uv, loss= self.args.eseploss)
            edgeseploss = torch.mean(torch.sum(edgeseploss, dim=2), dim=1)
            self.currentloss[self.count]['edgecorrespondences'] = edgecorrespondences

            # Relaxation
            if self.args.stitchrelax:
                edgeseploss = (edgeseploss * edgeseploss)/(edgeseploss * edgeseploss + seplossdelta)
            loss += self.args.stitchlossweight * torch.mean(edgeseploss)
            self.currentloss[self.count]['edgeseploss'] = edgeseploss.detach().cpu().numpy()

        if self.args.lossgradientstitching:
            if self.args.lossgradientstitching != 'split':
                edgegradloss, edgecorrespondences = uvgradloss(faces, uv, return_edge_correspondence=True, loss=self.args.lossgradientstitching)

                if self.args.lossgradientstitching in ['l1', 'l2']:
                    edgegradloss = torch.sum(edgegradloss, dim=1)

                self.currentloss[self.count]['edgecorrespondences'] = edgecorrespondences
            elif self.args.lossgradientstitching == "split":
                edgegradloss = splitgradloss(vertices, faces, uv, cosine_weight=1, mag_weight=1)

            if self.args.stitchrelax:
                edgegradloss = (edgegradloss * edgegradloss)/(edgegradloss * edgegradloss + seplossdelta)

            loss += self.args.stitchlossweight * torch.mean(edgegradloss)
            self.currentloss[self.count]['edgegradloss'] = edgegradloss.detach().cpu().numpy()

            # If transuv given, then also compute the edge separation loss
            if transuv is not None:
                edgeseploss = torch.sum(uvseparation(vertices, faces, transuv, loss='l2'), dim=[1,2])
                loss += self.args.seplossweight * torch.mean(edgeseploss)
                self.currentloss[self.count]['edgeseploss'] = edgeseploss.detach().cpu().numpy()

        self.currentloss[self.count]['total'] = loss.item()
        self.count += 1

        return loss

    def exportloss(self):
        return self.currentloss

# ==================== Energies ===============================
# Edge distortion: MSE between UV edge norm and original edge norm
def edgedistortion(vs, fs, uv):
    """vs: V x 3
       fs: F x 3
       uv: F x 3 x 2 """
    from meshing import Mesh
    from meshing.analysis import computeFacetoEdges

    # NOTE: Mesh data structure doesn't respect original face ordering
    # so we have to use custom edge-face connectivity export function
    mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    computeFacetoEdges(mesh)

    # Edge lengths per triangle
    f_elens = []
    uv_elens = []
    fverts = vs[fs] # F x 3 x 3
    for fi in range(len(fverts)):
        f_elens.append([torch.linalg.norm(fverts[fi, 1] - fverts[fi, 0]), torch.linalg.norm(fverts[fi, 2] - fverts[fi, 1]), torch.linalg.norm(fverts[fi, 2] - fverts[fi, 0])])
        uv_elens.append([torch.linalg.norm(uv[fi, 1] - uv[fi, 0]), torch.linalg.norm(uv[fi, 2] - uv[fi, 1]), torch.linalg.norm(uv[fi, 2] - uv[fi, 0])])

    f_elens = torch.tensor(f_elens, device=uv.device)
    uv_elens = torch.tensor(uv_elens, device=uv.device)
    energy = torch.nn.functional.mse_loss(uv_elens, f_elens, reduction='none')

    # Sum over triangles
    energy = torch.sum(energy, dim=1)

    return energy

# Autocuts energy involves weighted sum of two measures
#   - Triangle distortion: symmetric dirichlet (sum of Fnorm of jacobian and inverse jacobian)
#   - Edge separation: f(L1 norm between UVs of corresponding vertices) w/ f = x^2/(x^2 + delta) (delta smoothing parameter -> converge to 0 over time)
def autocuts(vs, fs, js, uv, sepweight=1, delta=0.01, init_j=None):
    """vs: V x 3
       fs: F x 3
       js: F x 3 x 2
       uv: F x 3 x 2 """
    dirichlet = torch.mean(symmetricdirichlet(vs, fs, uv, js, init_j))

    # Get vertex correspondences per edge
    # NOTE: ordering of vertices in UV MUST BE SAME as ordering of vertices in fs
    separation = uvseparation(vs, fs, uv)
    separation = torch.mean((separation * separation)/(separation * separation + delta))
    energy = dirichlet + sepweight * separation
    return energy

# TODO: batch this across multiple meshes
# NOTE: if given an initialization jacobian, then predicted jacob must be 2x2 matrix
def symmetricdirichlet(vs, fs, jacob=None, init_jacob=None):
    # Jacob: F x 2 x 3
    # TODO: Below can be precomputed
    # Get face areas
    from meshing import Mesh
    from meshing.analysis import computeFaceAreas
    mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    computeFaceAreas(mesh)
    fareas = torch.from_numpy(mesh.fareas).to(vs.device)

    if jacob is not None:
        mdim = 4 # NOTE: We will ALWAYS assume a 2x2 transformation so inverse is well defined
        if init_jacob is not None:
            # NOTE: assumes left multiplication => J' x J_I x Fvert
            # Use 2x2 upper diagonal of predicted jacobian
            if init_jacob.shape[1] > 2:
                init_jacob = init_jacob[:,:2,:]

            jacob2 = torch.matmul(jacob[:,:2,:2], init_jacob) # F x 2 x 3
            # jacob2 = torch.matmul(init_jacob, jacob[:,:2,:2]) # F x 2 x 2

            # Need final jacobian to be 2x2
            if jacob2.shape[2] > 2:
                jacob2 = torch.matmul(jacob2, jacob2.transpose(1,2))
        elif jacob.shape[2] > 2:
            # Map jacob to 2x2 by multiplying against transpose
            jacob2 = torch.matmul(jacob, jacob.transpose(1,2))
        else:
            jacob2 = jacob
        try:
            invjacob = torch.linalg.inv(jacob2)
        except Exception as e:
            print(f"Torch inv error on jacob2: {e}")
            invjacob = torch.linalg.pinv(jacob2)
        energy = fareas * (torch.sum((jacob2 * jacob2).reshape(-1, mdim), dim=-1) + torch.sum((invjacob * invjacob).reshape(-1, mdim), dim=-1) - 4)
    else:
        # Rederive jacobians from UV values
        raise NotImplementedError("Symmetric dirichlet with manual jacobian calculation not implemented yet!")
    return energy

def uvseparation(vs, fs, uv, loss='l1'):
    """ uv: F x 3 x 2
        vs: V x 3 (original topology)
        fs: F x 3 (original topology) """
    from source_njf.utils import edge_soup_correspondences
    from meshing import Mesh
    uvsoup = uv.reshape(-1, 2)
    edgecorrespondences, facecorrespondences = edge_soup_correspondences(fs.detach().cpu().numpy())
    e1 = []
    e2 = []
    for k, v in sorted(edgecorrespondences.items()):
        # If only one correspondence, then it is a boundary
        if len(v) == 1:
            continue
        e1.append(uvsoup[list(v[0])])
        e2.append(uvsoup[list(v[1])])

    ef0 = torch.stack(e1) # E*2 x 2
    ef1 = torch.stack(e2) # E*2 x 2

    # Weight each loss by length of 3D edge
    mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    elens = []
    for i, edge in sorted(mesh.topology.edges.items()):
        if edge.onBoundary():
            continue
        elens.append(mesh.length(edge))
    elens = torch.tensor(elens, device=uv.device).reshape(len(elens), 1,1)

    if loss == "l1":
        separation = elens * torch.nn.functional.l1_loss(ef0, ef1, reduction='none')
    elif loss == 'l2':
        separation = elens * torch.nn.functional.mse_loss(ef0, ef1, reduction='none')

    return separation, edgecorrespondences

def uvgradloss(fs, uv, return_edge_correspondence=False, loss='l2'):
    """ uv: F x 3 x 2
        vs: V x 3 (original topology)
        fs: F x 3 (original topology) """

    from source_njf.utils import edge_soup_correspondences
    # NOTE: Mesh data structure doesn't respect original face ordering
    # so we have to use custom edge-face connectivity export function
    # mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    # fconn, vconn = mesh.topology.export_edge_face_connectivity(mesh.faces)
    # fconn = np.array(fconn, dtype=int) # E x {f0, f1}
    # vconn = np.array(vconn, dtype=int) # E x {v0,v0'} x {v1, v1'}

    uvsoup = uv.reshape(-1, 2)
    edgecorrespondences, facecorrespondences = edge_soup_correspondences(fs.detach().cpu().numpy())
    e1 = []
    e2 = []
    for k, v in sorted(edgecorrespondences.items()):
        # If only one correspondence, then it is a boundary
        if len(v) == 1:
            continue
        e1.append(uvsoup[list(v[0])])
        e2.append(uvsoup[list(v[1])])

    ef0 = torch.cat(e1) # E*2 x 2
    ef1 = torch.cat(e2) # E*2 x 2

    # Debugging: visualize the edge vectors
    # import polyscope as ps
    # ps.init()
    # ps_uv = ps.register_surface_mesh("uv", uvsoup, np.arange(len(uvsoup)).reshape(-1, 3), edge_width=1)

    # # Map vconn to flattened triangle indices
    # # NOTE: FCONN AND VCONN NO GOOD REDO
    # vertcurve = np.arange(len(ef0)).reshape(-1, 2)
    # ecolors = np.arange(len(vertcurve))

    # ps_curve = ps.register_curve_network("edgeside1", ef0, vertcurve, enabled=True)
    # ps_curve.add_scalar_quantity("ecolors", ecolors, defined_on='edges', enabled=True)
    # ps_curve = ps.register_curve_network("edgeside2", ef1, vertcurve, enabled=True)
    # ps_curve.add_scalar_quantity("ecolors", ecolors, defined_on='edges', enabled=True)
    # ps.show()

    # Compare the edge vectors (just need to make sure the edge origin vertices are consistent)
    e0 = ef0[::2] - ef0[1::2] # E x 2
    e1 = ef1[::2] - ef1[1::2] # E x 2

    # Weight each loss by length of 3D edge
    # from meshing.mesh import Mesh
    # mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    # elens = []
    # for i, edge in sorted(mesh.topology.edges.items()):
    #     if edge.onBoundary():
    #         continue
    #     elens.append(mesh.length(edge))
    # elens = torch.tensor(elens, device=uv.device).reshape(len(elens), 1)
    # elens /= torch.max(elens)

    if loss == "l1":
        separation = torch.nn.functional.l1_loss(e0, e1, reduction='none')
    elif loss == 'l2':
        separation = torch.nn.functional.mse_loss(e0, e1, reduction='none')
    elif loss == 'cosine':
        # Cosine similarity
        separation = 1 - torch.nn.functional.cosine_similarity(e0, e1, eps=1e-8)

    if return_edge_correspondence:
        return separation, edgecorrespondences

    return separation

def splitgradloss(vs, fs, uv, cosine_weight=1, mag_weight=1):
    """ uv: F x 3 x 2
        vs: V x 3 (original topology)
        fs: F x 3 (original topology) """

    from meshing import Mesh
    # NOTE: Mesh data structure doesn't respect original face ordering
    # so we have to use custom edge-face connectivity export function
    mesh = Mesh(vs.detach().cpu().numpy(), fs.detach().cpu().numpy())
    fconn, vconn = mesh.topology.export_edge_face_connectivity(mesh.faces)
    fconn = np.array(fconn, dtype=int) # E x {f0, f1}
    vconn = np.array(vconn, dtype=int) # E x {v0,v1}
    ef0 = uv[fconn[:,[0]], vconn[:,0]] # E x 2 x 2
    ef1 = uv[fconn[:,[1]], vconn[:,1]] # E x 2 x 2

    # Compare the edge vectors (just need to make sure the edge origin vertices are consistent)
    e0 = ef0[:,1] - ef0[:,0]
    e1 = ef1[:,1] - ef1[:,0]

    # Weight each loss by length of 3D edge
    elens = []
    for i, edge in sorted(mesh.topology.edges.items()):
        if edge.onBoundary():
            continue
        elens.append(mesh.length(edge))
    elens = torch.tensor(elens, device=uv.device)

    cosine_loss = -torch.nn.functional.cosine_similarity(e0, e1)
    mag_loss = torch.nn.functional.mse_loss(torch.norm(e0, dim=1)/elens, torch.norm(e1, dim=1)/elens, reduction='none')

    return elens * (cosine_weight * cosine_loss + mag_weight * mag_loss)

def arap(local_tris, faces, param=None, return_face_energy=True, paramtris=None, renormalize=True,
         face_weights=None, normalize_filter=0, device=torch.device("cpu"), verbose=False, timeit=False, **kwargs):
    if paramtris is None:
        paramtris = param[faces]

    if timeit == True:
        import time
        t0 = time.time()

    # Squared norms of difference in edge vectors multiplied by cotangent of opposite angle
    # NOTE: LSCM applies some constant scaling factor -- can we renormalize to get back original edge lengths?
    try:
        local_tris = local_tris.contiguous()
    except Exception as e:
        print(e)

    e1 = local_tris[:, 2, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e3 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e1_p = paramtris[:, 2, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e3_p = paramtris[:, 2, :] - paramtris[:, 1, :]

    # NOTE: sometimes denominator will be 0 i.e. area of triangle is 0 -> cotangent in this case is infty, default to 1e5
    cot1 = torch.abs(torch.sum(e2 * e3, dim=1) / torch.clamp((e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]), min=1e-3))
    cot2 = torch.abs(torch.sum(e1 * e3, dim=1) / torch.clamp((e1[:, 0] * e3[:, 1] - e1[:, 1] * e3[:, 0]), min=1e-3))
    cot3 = torch.abs(torch.sum(e2 * e1, dim=1) / torch.clamp(e2[:, 0] * e1[:, 1] - e2[:, 1] * e1[:, 0], min=1e-3))

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Threshold param tris as well
    e1_p = torch.maximum(torch.minimum(e1_p, torch.tensor(1e5)), torch.tensor(-1e5))
    e2_p = torch.maximum(torch.minimum(e2_p, torch.tensor(1e5)), torch.tensor(-1e5))
    e3_p = torch.maximum(torch.minimum(e3_p, torch.tensor(1e5)), torch.tensor(-1e5))

    # Compute all edge rotations
    rot_e1 = []
    rot_e2 = []
    rot_e3 = []
    for i in range(len(e1)):
        crosscov = cot1[i] * torch.mm(e1[i].unsqueeze(0).t(), e1_p[i].unsqueeze(0)) + \
                   cot2[i] * torch.mm(e2[i].unsqueeze(0).t(), e2_p[i].unsqueeze(0)) + \
                   cot3[i] * torch.mm(e3[i].unsqueeze(0).t(), e3_p[i].unsqueeze(0))
        a, b, c, d = crosscov.flatten()
        E = (a + d) / 2
        F = (a - d) / 2
        G = (c + b) / 2
        H = (c - b) / 2
        Q = torch.sqrt(E ** 2 + H ** 2)
        R = torch.sqrt(F ** 2 + G ** 2)
        S1 = Q + R
        S2 = Q - R
        a1 = torch.atan2(G, F)
        a2 = torch.atan2(H, E)
        theta = (a2 - a1) / 2
        phi = (a2 + a1) / 2
        U = torch.tensor([[torch.cos(phi), -torch.sin(phi)], [torch.sin(phi), torch.cos(phi)]])
        S = torch.tensor([S1, S2])
        V = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
        #
        # theta = 0.5 * torch.atan2(2*a*c + 2*b*d, a**2 + b**2 - c**2 - d**2)
        # U = torch.tensor([[torch.cos(theta), -torch.sin(theta)],[torch.sin(theta), torch.cos(theta)]])
        # S1 = a**2 + b**2 + c**2 + d**2
        # S2 = torch.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4*(a*c+b*d)**2)
        # S = torch.tensor([S1, S2])
        # phi = 0.5 * torch.atan2(2*a*b + 2*c*d, a**2 - b**2 + c**2 - d**2)
        # s11 = (a * torch.cos(theta) + c * torch.sin(theta)) * torch.cos(phi) + \
        #       (b * torch.cos(theta) + d*torch.sin(theta))*torch.sin(phi)
        # s22 = (a * torch.sin(theta) - c * torch.cos(theta)) * torch.sin(phi) + \
        #       (-b * torch.sin(theta) + d*torch.cos(theta))*torch.cos(phi)
        # V = torch.t(torch.tensor([[torch.sign(s11) * torch.cos(phi), -torch.sign(s22)*torch.sin(phi)],
        #                   [torch.sign(s11) * torch.sin(phi), torch.sign(s22)*torch.cos(phi)]]))
        # U_test, S_test, V_test = torch.linalg.svd(crosscov)
        # Make sure that determinant of rotation matrix is positive
        R = torch.mm(torch.t(V), torch.t(U)).to(device)
        if torch.det(R) <= 0:
            U[:, 1] *= -1
            R = torch.mm(torch.t(V), torch.t(U)).to(device)
        # print(f"Computed SVD")
        # print(U)
        # print(S)
        # print(V)
        # print(f"Recomputed matrix: {U @ torch.diag(S) @ V}")
        # print(f"Pytorch SVD")
        # print(U_test)
        # print(S_test)
        # print(V_test)
        # print(f"Recomputed matrix: {U_test @ torch.diag(S_test) @ V_test}")

        edge_tmp = torch.column_stack([e1[i], e2[i], e3[i]]).to(device)
        rot_edges = torch.mm(R, edge_tmp)
        rot_e1.append(rot_edges[:, 0])
        rot_e2.append(rot_edges[:, 1])
        rot_e3.append(rot_edges[:, 2])

        # print("Param E1:", e1_p[i])
        # print("Param E2:", e2_p[i])
        # print("Param E3:", e3_p[i])
        # print("Rot E1:", rot_edges[:,0])
        # print("Rot E2:", rot_edges[:,1])
        # print("Rot E3:", rot_edges[:,2])
        # print("OG E1:", e1[i])
        # print("OG E2:", e2[i])
        # print("OG E3:", e3[i])
        #
        # # Get angles
        # e1_angle_old = torch.acos(torch.dot(e1_p[i], e1[i])/(torch.linalg.norm(e1_p[i]) * torch.linalg.norm(e1[i])))
        # e2_angle_old = torch.acos(torch.dot(e2_p[i], e2[i])/(torch.linalg.norm(e2_p[i]) * torch.linalg.norm(e2[i])))
        # e3_angle_old = torch.acos(torch.dot(e3_p[i], e3[i])/(torch.linalg.norm(e3_p[i]) * torch.linalg.norm(e3[i])))
        # e1_angle = torch.acos(torch.dot(e1_p[i], rot_edges[:,0])/(torch.linalg.norm(e1_p[i]) * torch.linalg.norm(rot_edges[:,0])))
        # e2_angle = torch.acos(torch.dot(e2_p[i], rot_edges[:,1])/(torch.linalg.norm(e2_p[i]) * torch.linalg.norm(rot_edges[:,1])))
        # e3_angle = torch.acos(torch.dot(e3_p[i], rot_edges[:,2])/(torch.linalg.norm(e3_p[i]) * torch.linalg.norm(rot_edges[:,2])))
        # print(f"Pre-rotation angle difference: {e1_angle_old}, {e2_angle_old}, {e3_angle_old}")
        # print(f"Post-rotation angle difference: {e1_angle}, {e2_angle}, {e3_angle}")
        # exit()

    rot_e1 = torch.stack(rot_e1)
    rot_e2 = torch.stack(rot_e2)
    rot_e3 = torch.stack(rot_e3)

    # print(f"Cot1: {cot1[:2]}")
    # print(f"Cot2: {cot2[:2]}")
    # print(f"Cot3: {cot3[:2]}")
    # print(f"rot_e1: {rot_e1[:2]}")
    # print(f"rot_e2: {rot_e2[:2]}")
    # print(f"rot_e3: {rot_e3[:2]}")
    # print(f"e1_p: {e1_p[:2]}")
    # print(f"e2_p: {e2_p[:2]}")
    # print(f"e3_p: {e3_p[:2]}")
    # print(f"e1: {e1[:2]}")
    # print(f"e2: {e2[:2]}")
    # print(f"e3: {e3[:2]}")

    if renormalize == True:
        # ARAP-minimizing scaling of parameterization edge lengths
        num = 0
        denom = 0
        for i in range(len(rot_e1)):
            if face_weights is not None:
                if face_weights[i] <= normalize_filter:
                    continue
            num += cot1[i] * torch.dot(rot_e1[i], e1_p[i]) + \
                   cot2[i] * torch.dot(rot_e2[i], e2_p[i]) + \
                   cot3[i] * torch.dot(rot_e3[i], e3_p[i])
            denom += (cot1[i] * torch.dot(e1_p[i], e1_p[i]) +
                      cot2[i] * torch.dot(e2_p[i], e2_p[i]) +
                      cot3[i] * torch.dot(e3_p[i], e3_p[i]))

            # Debugging: non-finite values
            # assert torch.isfinite(num), f"Non-finite value encountered in numerator: cot1 = {cot1[i]}" \
            #                             f", cot2 = {cot2[i]}, cot3 = {cot3[i]}, rot_e1 = {rot_e1[i]}, rot_e2 = {rot_e2[i]}" \
            #                             f", rot_e3 = {rot_e3[i]}, e1_p = {e1_p[i]}, e2_p = {e2_p[i]}, e3_p = {e3_p[i]}"
            # assert torch.isfinite(denom), f"Non-finite value encountered in denominator: cot1 = {cot1[i]}" \
            #                             f", cot2 = {cot2[i]}, cot3 = {cot3[i]}, " \
            #                             f"e1_p = {e1_p[i]}, e2_p = {e2_p[i]}, e3_p = {e3_p[i]}"

            # print(f"num: {num}, denum: {denom}")
            # print(cot1[i], cot2[i], cot3[i])
        ratio = max(num / denom, 1e-5)
        if verbose == True:
            print(f"Scaling param. edges by ARAP-minimizing scalar: {ratio}")
        e1_p_norm = ratio * e1_p
        e2_p_norm = ratio * e2_p
        e3_p_norm = ratio * e3_p

    # If any non-finite values, then return None
    if not torch.all(torch.isfinite(e1_p)) or not torch.all(torch.isfinite(e2_p)) or not torch.all(torch.isfinite(e3_p)) or \
        not torch.all(torch.isfinite(rot_e1)) or not torch.all(torch.isfinite(rot_e2)) or not torch.all(torch.isfinite(rot_e3)):
        print(f"ARAP: non-finite elements found")
        return None

    # Compute face-level distortions
    arap_tris = []
    for i in range(len(e1)):
        if renormalize == True:
            arap_tmp = cot1[i] * torch.linalg.norm(e1_p_norm[i] - rot_e1[i]) ** 2 + \
                       cot2[i] * torch.linalg.norm(e2_p_norm[i] - rot_e2[i]) ** 2 + \
                       cot3[i] * torch.linalg.norm(e3_p_norm[i] - rot_e3[i]) ** 2
        else:
            arap_tmp = cot1[i] * torch.linalg.norm(e1_p[i] - rot_e1[i]) ** 2 + \
                       cot2[i] * torch.linalg.norm(e2_p[i] - rot_e2[i]) ** 2 + \
                       cot3[i] * torch.linalg.norm(e3_p[i] - rot_e3[i]) ** 2
        arap_tris.append(arap_tmp)

        # Debugging
        # arap_tmp_old = cot1[i] * torch.linalg.norm(e1_p_old[i] - rot_e1[i]) ** 2 + \
        #            cot2[i] * torch.linalg.norm(e2_p_old[i] - rot_e2[i]) ** 2 + \
        #            cot3[i] * torch.linalg.norm(e3_p_old[i] - rot_e3[i]) ** 2
        # print(f"Pre scaling distortion: {arap_tmp_old}")
        # print(f"Post scaling distortion: {arap_tmp}")
        # exit()
    arap = torch.stack(arap_tris)

    if timeit == True:
        print(f"ARAP calculation: {time.time()-t0:0.5f}")

    if return_face_energy == False:
        return torch.mean(arap)
    return arap

# Smoothness term for graphcuts: basically sum pairwise energies as function of label pairs + some kind of geometry weighting
# pairwise: maximum cost of diverging neighbor labels (i.e. 0 next to 1). this value is multiplied by the difference in the soft probs
def gcsmoothness(preds, mesh, feature='dihedral', pairwise=1):
    face_adj = torch.tensor([[edge.halfedge.face.index, edge.halfedge.twin.face.index] for key, edge in sorted(mesh.topology.edges.items())]).long().to(preds.device)

    if feature == 'dihedral':
        if not hasattr(mesh, "dihedrals"):
            from models.layers.meshing.analysis import computeDihedrals
            computeDihedrals(mesh)

        dihedrals = torch.clip(torch.pi - torch.from_numpy(mesh.dihedrals).to(preds.device), 0, torch.pi).squeeze()

        # TODO: maybe send all dihedrals past 90* to smoothness cost 0
        # Maps dihedrals from 0 => infty
        # smoothness = -torch.log(dihedrals/(torch.pi - dihedrals + 1e-8) + 1e-8)
        smoothness = -torch.log(dihedrals/torch.pi + 1e-15)
    else:
        raise NotImplementedError(feature)

    adj_preds = preds[face_adj]

    # NOTE: we include the 1 - torch.max() term in order to encourage patch GROWING
    smoothness_cost = torch.mean(smoothness * pairwise * ((torch.abs(adj_preds[:,1] - adj_preds[:,0]))))

    return smoothness_cost

def batchedgcsmoothness(preds, meshes, feature='dihedral', pairwise=10):
    face_adj = torch.tensor([[edge.halfedge.face.index, edge.halfedge.twin.face.index] for key, edge in sorted(mesh.topology.edges.items())]).long().to(preds.device)

    if feature == 'dihedral':
        if not hasattr(mesh, "dihedrals"):
            from models.layers.meshing.analysis import computeDihedrals
            computeDihedrals(mesh)

        dihedrals = torch.clip(torch.pi - torch.from_numpy(mesh.dihedrals).to(preds.device), 0, torch.pi).squeeze()

        # Maps dihedrals from 0 => infty
        smoothness = -torch.log(dihedrals/torch.pi + 1e-15)
    else:
        raise NotImplementedError(feature)

    # TODO: data cost (compare preds with rounded maybe) + label cost (apply high cost to only seeing one label)

    adj_preds = preds[face_adj]
    smoothness_cost = torch.mean(smoothness * pairwise * (torch.abs(adj_preds[:,1] - adj_preds[:,0])))

    return smoothness_cost

def arap_v2(local_tris, faces, param, return_face_energy=True, paramtris=None, renormalize=True,
         face_weights=None, normalize_filter=0, device=torch.device("cpu"), verbose=False, timeit=False, **kwargs):
    if paramtris is None:
        paramtris = param[faces]

    if timeit == True:
        import time
        t0 = time.time()

    # Squared norms of difference in edge vectors multiplied by cotangent of opposite angle
    # NOTE: LSCM applies some constant scaling factor -- can we renormalize to get back original edge lengths?
    try:
        local_tris = local_tris.contiguous()
    except Exception as e:
        print(e)

    e1 = local_tris[:, 2, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e3 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e1_p = paramtris[:, 2, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e3_p = paramtris[:, 2, :] - paramtris[:, 1, :]

    # NOTE: sometimes denominator will be 0 i.e. area of triangle is 0 -> cotangent in this case is infty, default to 1e5
    cot1 = torch.abs(torch.sum(e2 * e3, dim=1) / torch.clamp((e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]), min=1e-5))
    cot2 = torch.abs(torch.sum(e1 * e3, dim=1) / torch.clamp((e1[:, 0] * e3[:, 1] - e1[:, 1] * e3[:, 0]), min=1e-5))
    cot3 = torch.abs(torch.sum(e2 * e1, dim=1) / torch.clamp(e2[:, 0] * e1[:, 1] - e2[:, 1] * e1[:, 0], min=1e-5))

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Threshold param tris as well
    e1_p = torch.maximum(torch.minimum(e1_p, torch.tensor(1e5)), torch.tensor(-1e5))
    e2_p = torch.maximum(torch.minimum(e2_p, torch.tensor(1e5)), torch.tensor(-1e5))
    e3_p = torch.maximum(torch.minimum(e3_p, torch.tensor(1e5)), torch.tensor(-1e5))

    # Compute all edge rotations
    cot_full = torch.stack([cot1, cot2, cot3]).reshape(3, len(cot1), 1, 1)
    e_full = torch.stack([e1, e2, e3])
    e_p_full = torch.stack([e1_p, e2_p, e3_p])
    crosscov = torch.sum(cot_full * torch.matmul(e_full.unsqueeze(3), e_p_full.unsqueeze(2)), dim=0)
    crosscov = crosscov.reshape(crosscov.shape[0], 4) # F x 4

    # tdenom = torch.clamp(crosscov[:,0]**2 + crosscov[:,1]**2 - crosscov[:,2]**2 - crosscov[:,3]**2, min=1e-5)
    # pdenom = torch.clamp(crosscov[:,0]**2 - crosscov[:,1]**2 + crosscov[:,2]**2 - crosscov[:,3]**2, min=1e-5)
    # theta = torch.atan2(2 * crosscov[:,0] * crosscov[:,2] + 2 * crosscov[:,1] * crosscov[:,3], tdenom)/2
    # phi = torch.atan2(2 * crosscov[:,0] * crosscov[:,1] + 2 * crosscov[:,2] * crosscov[:,3], pdenom)/2

    # cphi = torch.cos(phi)
    # sphi = torch.sin(phi)
    # ctheta = torch.cos(theta)
    # stheta = torch.sin(theta)
    # s1 = () * + () *
    # s2 = () * + () *

    # U = torch.stack([torch.stack([ctheta, -stheta], dim=1), torch.stack([stheta, ctheta], dim=1)], dim=2)
    # V = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1), torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=2)

    E = (crosscov[:,0] + crosscov[:,3])/2
    F = (crosscov[:,0] - crosscov[:,3])/2
    G = (crosscov[:,2] + crosscov[:,1])/2
    H = (crosscov[:,2] - crosscov[:,1])/2

    Q = torch.sqrt(E ** 2 + H ** 2)
    R = torch.sqrt(F ** 2 + G ** 2)

    S1 = Q + R
    S2 = Q - R
    a1 = torch.atan2(G, torch.clamp(F, min=1e-5))
    a2 = torch.atan2(H, torch.clamp(E, min=1e-5))
    theta = (a2 - a1) / 2 # F
    phi = (a2 + a1) / 2 # F

    # F x 2 x 2
    U = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1), torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=2)

    # F x 2 x 2
    V = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1), torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)], dim=2)

    R = torch.matmul(V, U).to(device) # F x 2 x 2
    baddet = torch.where(torch.det(R) <= 0)[0]
    if len(baddet) > 0:
        U[baddet, 1, :] *= -1
        R = torch.matmul(V, U).to(device)

    edge_tmp = torch.stack([e1, e2, e3], dim=2)
    rot_edges = torch.matmul(R, edge_tmp) # F x 2 x 3
    rot_e_full = rot_edges.permute(2, 0, 1) # 3 x F x 2
    cot_full = cot_full.reshape(cot_full.shape[0], cot_full.shape[1]) # 3 x F
    if renormalize == True:
        # ARAP-minimizing scaling of parameterization edge lengths
        if face_weights is not None:
            keepfs = torch.where(face_weights > normalize_filter)[0]
        else:
            keepfs = torch.arange(rot_e_full.shape[1])

        num = torch.sum(cot_full[:,keepfs] * torch.sum(rot_e_full[:,keepfs,:] * e_p_full[:,keepfs,:], dim = 2))
        denom = torch.sum(cot_full[:,keepfs] * torch.sum(e_p_full[:,keepfs,:] * e_p_full[:,keepfs,:], dim = 2))

        ratio = max(num / denom, 1e-5)
        if verbose == True:
            print(f"Scaling param. edges by ARAP-minimizing scalar: {ratio}")

        e_p_full *= ratio

    # If any non-finite values, then return None
    if not torch.all(torch.isfinite(e_p_full)) or not torch.all(torch.isfinite(rot_e_full)):
        print(f"ARAP: non-finite elements found")
        return None

    # Compute face-level distortions
    arap_tris = torch.sum(cot_full * torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2, dim=0)
    if timeit == True:
        print(f"ARAP calculation: {time.time()-t0:0.5f}")

    # Debugging: show rotated edges along with parameterization
    # import polyscope as ps
    # ps.init()
    # f1 = faces[0]
    # param_f1 = param[f1]
    # # Normalize the param so first vertex is at 0,0
    # param_f1 = param_f1 - param_f1[0]
    # og_f1 = local_tris[0] # 3 x 2
    # rot_f1 = R[0]
    # new_f1 = torch.matmul(rot_f1, og_f1.transpose(1,0)).transpose(1,0)
    # print(new_f1)
    # og_curve = ps.register_curve_network("og triangle", og_f1.numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[0,1,0])
    # param_curve = ps.register_curve_network("UV", param_f1.numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[0,0,1])
    # rot_curve = ps.register_curve_network("rot triangle", new_f1.numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[1,0,0])
    # ps.show()

    # # Compute energies
    # print(e_p_full.shape)
    # print(e_full.shape)
    # print(arap_tris[0])
    # print(torch.sum(cot_full[:,0] * torch.linalg.norm(e_p_full[:,0,:] - e_full[:,0,:], dim=1) ** 2))

    # raise

    if return_face_energy == False:
        return torch.mean(arap_tris)

    return arap_tris

# ==================== Loss Functions ===============================
# OG Counting Loss
def count_loss(face_errors, fareas, threshold=0.1, alpha=5, debug=False,
               return_softloss=True, device=torch.device("cpu"), **kwargs):
    # Normalize face error by parea
    error = face_errors
    fareas = fareas / torch.sum(fareas)
    softloss = fareas * (1 - torch.exp(-(error / threshold) ** alpha))
    count_loss = torch.sum(softloss)

    if debug == True:
        print(f"Error quantile: {torch.quantile(error, torch.linspace(0, 1, 5).to(device).double())}")
        print(
            f"Thresh loss quantile: {torch.quantile(torch.exp(-(error / threshold) ** alpha), torch.linspace(0, 1, 5).to(device).double())}")

    if return_softloss == True:
        return count_loss, softloss
    return count_loss

# Count loss with critical point at the threshold
def count_loss_v2(face_errors, fareas, threshold=0.1, alpha=5, debug=False,
               return_softloss=True, device=torch.device("cpu"), **kwargs):
    # Normalize face error by parea
    error = face_errors
    fareas = fareas / torch.sum(fareas)
    softloss = fareas * (torch.exp(-((error - threshold) / threshold) ** alpha) - torch.exp(-torch.ones(1)).to(error.device))
    count_loss = torch.sum(softloss)

    if debug == True:
        print(f"Error quantile: {torch.quantile(error, torch.linspace(0, 1, 5).to(device).double())}")
        print(
            f"Thresh loss quantile: {torch.quantile(torch.exp(-(error / threshold) ** alpha), torch.linspace(0, 1, 5).to(device).double())}")

    if return_softloss == True:
        return count_loss, softloss
    return count_loss

# Counting loss with uniformity loss
def unif_count_loss(mesh, weights, face_errors, fareas, threshold=0.1, alpha=2, debug=False,
                    return_softloss=True, device=torch.device("cpu"), return_components=False, **kwargs):
    error = face_errors
    softloss = fareas * (1 - torch.exp(-(error / threshold) ** alpha))
    count_loss = torch.sum(softloss) / torch.sum(fareas)  # Weighted average
    dihedrals = torch.from_numpy(np.array(mesh.face_adjacency_angles)).to(device)
    dihed_weights = weights[torch.from_numpy(np.array(mesh.face_adjacency)).to(device).long()]
    # dihed_loss = torch.mean(torch.abs(dihed_weights[:,1] - dihed_weights[:,0])/
    #                         (torch.exp((dihedrals/(torch.tensor(np.pi)-dihedrals)) ** gamma)))
    dihed_loss = torch.mean(torch.abs(dihed_weights[:, 1] - dihed_weights[:, 0]) / torch.exp(dihedrals))

    if debug == True:
        print(f"Count loss: {count_loss}")
        print(f"Uniformity loss: {dihed_loss}")

    # print(f"Count loss: {count_loss}")
    # print(f"Uniformity loss: {dihed_loss}")
    if return_softloss == True:
        return count_loss + dihed_loss, softloss
    elif return_components == True:
        return count_loss, dihed_loss
    return count_loss + dihed_loss


# Discourage UVs way out of bounds (LSCM can scale in really weird ways)
def uv_loss(uvs, l=1.0):
    loss = l * torch.mean(torch.tanh(torch.clamp(uvs.flatten() - 1, min=0) + torch.clamp(0 - uvs.flatten(), min=0)))
    return loss


# Encourages similar weights for faces with low dihedrals
def uniformity_loss(mesh, weights, device=torch.device("cpu")):
    dihedrals = torch.from_numpy(np.array(mesh.face_adjacency_angles)).to(device)
    dihed_weights = weights[torch.from_numpy(np.array(mesh.face_adjacency)).to(device).long()]
    dihed_loss = torch.mean(torch.abs(dihed_weights[:, 1] - dihed_weights[:, 0]) / torch.exp(dihedrals))
    return dihed_loss


# Counting loss using sigmoid (maybe better gradients)
def count_loss2(face_errors, fareas, threshold=0.1, alpha=1, device=torch.device("cpu")):
    softloss = fareas * torch.sigmoid(((face_errors - threshold) / threshold) ** alpha)
    count_loss = torch.sum(softloss) / torch.sum(fareas)  # Weighted average by face area
    return count_loss


def count_loss3(face_errors, fareas, threshold=0.1, alpha=1, device=torch.device("cpu")):
    softloss = fareas * (torch.tanh(torch.tensor(-1)) + torch.tanh(((face_errors - threshold) / threshold) ** alpha))
    count_loss = torch.sum(softloss) / torch.sum(fareas)  # Weighted average by face area
    return count_loss


# Counting loss with uniformity loss
def unif_count_loss2(face_adjacency, dihedrals, weights, face_errors, fareas, threshold=0.1, alpha=2, debug=False,
                     return_softloss=True, device=torch.device("cpu"), return_components=False, **kwargs):
    error = face_errors
    softloss = fareas * (1 - torch.exp(-(error / threshold) ** alpha))
    count_loss = torch.sum(softloss) / torch.sum(fareas)  # Weighted average
    dihedrals = dihedrals.to(device)
    dihed_weights = weights[face_adjacency.to(device).long()]
    dihed_loss = torch.mean(torch.abs(dihed_weights[:, 1] - dihed_weights[:, 0]) / torch.exp(dihedrals))

    if debug == True:
        print(f"Count loss: {count_loss}")
        print(f"Uniformity loss: {dihed_loss}")

    # print(f"Count loss: {count_loss}")
    # print(f"Uniformity loss: {dihed_loss}")
    if return_softloss == True:
        return count_loss + dihed_loss, softloss
    elif return_components == True:
        return count_loss, dihed_loss
    return count_loss + dihed_loss


# Count loss with stretch penalty
# def stretch_count_loss(face_errors, pareas, fareas, threshold = 0.1, alpha  = 2, debug = False,
#                         return_softloss=True, device = torch.device("cpu"), **kwargs):
#     fareas = fareas/torch.sum(fareas)

#     # Normalize face error by parea
#     error = face_errors/pareas
#     stretch = (pareas/fareas + fareas/pareas)
#     # print(torch.quantile(angle_loss.float(), torch.linspace(0,1,5).to(device)))
#     # print(torch.where(error == 0))
#     softloss = stretch * fareas * (1 - torch.exp(-(error/threshold)**alpha))
#     count_loss = torch.sum(softloss)
#     if return_softloss==True:
#         return count_loss, softloss
#     return count_loss

# Weighted distortion loss with uniformity
def weight_loss(mesh, weights, face_errors, gamma=0.5, alpha=1, device=torch.device("cpu"),
                return_softloss=True, return_components=False, debug=False, **kwargs):
    error = face_errors
    # distort_loss = torch.mean(error)
    distort_loss = torch.mean(1 - torch.exp(-error * weights))

    dihedrals = torch.from_numpy(np.array(mesh.face_adjacency_angles)).to(device)
    dihed_weights = weights[torch.from_numpy(np.array(mesh.face_adjacency)).to(device).long()]
    dihed_loss = torch.mean(torch.abs(dihed_weights[:, 1] - dihed_weights[:, 0]) / torch.exp(dihedrals))
    weight_loss = 1 - torch.mean(weights)

    # print(f"Distort loss: {distort_loss}")
    # print(f"Uniformity loss: {dihed_loss}")
    # print(f"Weight loss: {weight_loss}")

    if return_softloss == True:
        return distort_loss + dihed_loss + weight_loss, error
    elif return_components == True:
        return distort_loss, dihed_loss, weight_loss
    return distort_loss + dihed_loss


# Face-errors normalized by face areas
def norm_weight_loss(mesh, weights, face_errors, fareas, gamma=0.5, alpha=1, debug=False,
                     return_softloss=True, device=torch.device("cpu"), **kwargs):
    error = face_errors
    fareas = fareas / torch.sum(fareas)
    softloss = fareas * error
    distort_loss = torch.sum(softloss)

    dihedrals = torch.from_numpy(np.array(mesh.face_adjacency_angles)).to(device)
    dihed_weights = weights[torch.from_numpy(np.array(mesh.face_adjacency)).to(device).long()]
    dihed_loss = torch.mean(torch.abs(dihed_weights[:, 1] - dihed_weights[:, 0]) / torch.exp(dihedrals))

    if debug == True:
        print(f"Distort loss: {distort_loss}")
        print(f"Uniformity loss: {dihed_loss}")

    if return_softloss == True:
        return distort_loss + dihed_loss, softloss
    return distort_loss + dihed_loss


# ================ Classification Metrics =====================
def compute_pr_auc(labels, preds):
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    auc_score = auc(recall, precision)
    return precision, recall, thresholds, auc_score

def auc(labels, preds):
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(labels, preds)
    return auc_score

def mAP(labels, preds, multi=False):
    from sklearn.metrics import average_precision_score
    if multi == False:
        return average_precision_score(labels, preds)
    else:
        map = 0.0
        for i in range(len(labels)):
            label = labels[i]
            pred = preds[i]
            map += average_precision_score(label, pred)
        return map/i

def f1(labels, preds, multi=False):
    from sklearn.metrics import f1_score
    if multi == False:
        return f1_score(labels, preds)
    else:
        f1 = 0.0
        for i in range(len(labels)):
            label = labels[i]
            pred = preds[i]
            f1 += f1_score(label, pred)
        return f1 / i