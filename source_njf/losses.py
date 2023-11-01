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
        # assert not (self.args.lossedgeseparation and self.args.lossgradientstitching), f"Can only have one type of edge loss!"

        # Record loss names (for visualization)
        self.lossnames = []

        if self.args.stitchingloss is not None:
            for lossname in self.args.stitchingloss:
                self.lossnames.append(lossname)

        if self.args.lossdistortion:
            self.lossnames.append("distortionloss")

    def clear(self):
        self.currentloss = defaultdict(dict)
        self.count = 0

    def computeloss(self, vertices = None, faces = None, uv = None, jacobians = None, initjacobs=None, seplossdelta=0.1,
                    weights = None, stitchweights = None, source = None, keepidxs = None):
        loss = 0
        # Ground truth
        if self.args.gtuvloss:
            # TODO: DOUBLE CHECK THAT VERTEX ORDER WITHIN FACES IS CORRECT OTHERWISE THIS MISMATCHES
            gtuvloss = torch.nn.functional.mse_loss(uv.reshape(-1, 2), source.get_loaded_data('gt_uvs'), reduction='none')
            loss += torch.sum(gtuvloss)
            self.currentloss[self.count]['gtuvloss'] = np.mean(gtuvloss.detach().cpu().numpy(), axis=1)

        # Shrinking penalty
        if self.args.invjloss:
            # Penalize sqrt(2)/||J||^2 + epsilon
            jnorm = torch.linalg.norm(jacobians, ord='fro', dim=(2,3))**2
            invjloss = torch.sqrt(torch.tensor(2))/(jnorm.squeeze() + 1e-5)
            loss += torch.mean(invjloss)
            self.currentloss[self.count]['invjloss'] = invjloss.detach().cpu().numpy()

        # Autocuts
        # if self.args.lossautocut:
        #     acloss = autocuts(vertices, faces, jacobians, uv, self.args.seplossweight, seplossdelta)
        #     loss += acloss
        #     self.currentloss[self.count]['autocuts'] = acloss.detach().item()

        ## Stitching loss: can be vertex separation, edge separation, or gradient stitching
        ## Options: {l1/l2 distance}, {seamless transformation}, {weighting}
        if self.args.stitchingloss is not None:
            edgesep, stitchingdict, weightdict = stitchingloss(vertices, faces, uv.reshape(-1, 2), self.args.stitchingloss, self.args,
                                                      stitchweights=stitchweights, source = source,
                                                      keepidxs = keepidxs)
            for k, v in stitchingdict.items():
                self.currentloss[self.count][k] = v.detach().cpu().numpy()
                loss += weightdict[k] * torch.sum(v)

            # Edge sep always goes into lossdict for visualization purposes
            self.currentloss[self.count]['edgeseparation'] = edgesep.detach().cpu().numpy()

        ### Distortion loss: can be ARAP or symmetric dirichlet
        distortionenergy = None
        if self.args.lossdistortion == "arap":
            distortionenergy = arap(vertices, faces, uv, paramtris=uv,
                                device=self.device, renormalize=False,
                                return_face_energy=True, timeit=False)
            self.currentloss[self.count]['distortionloss'] = distortionenergy.detach().cpu().numpy()

            if not self.args.losscount:
                loss += self.args.distortion_weight * torch.sum(distortionenergy)

        if self.args.lossdistortion == "dirichlet":
            distortionenergy = symmetricdirichlet(vertices, faces, jacobians.squeeze(), init_jacob=initjacobs)
            self.currentloss[self.count]['distortionloss'] = distortionenergy.detach().cpu().numpy()

            if not self.args.losscount:
                loss += self.args.distortion_weight * torch.sum(distortionenergy)

        # Cut sparsity loss
        if self.args.sparsecutsloss:
            sparseloss = parabolaloss(weights).mean()
            self.currentloss[self.count]['sparsecutsloss'] = sparseloss.detach().cpu().numpy()
            loss += self.args.sparsecuts_weight * sparseloss

        self.currentloss[self.count]['total'] = loss.item()
        self.count += 1

        return loss

    def exportloss(self):
        return self.currentloss

# ==================== Edge Cut Sparsity Losses ===============================
def parabolaloss(weights):
    """ Penalizes weights at 0.5 and none at 0/1. Min 0 and max 1 assuming weights are in [0,1]. """
    return -4 * (weights - 0.5) ** 2 + 1

# ==================== Distortion Energies ===============================
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
        # NOTE: This assumes jacob matrix is size B x J1 x J2
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

def arap(vertices, faces, param, return_face_energy=True, paramtris=None, renormalize=False,
         face_weights=None, normalize_filter=0, device=torch.device("cpu"), verbose=False, timeit=False, **kwargs):
    from source_njf.utils import get_local_tris
    local_tris = get_local_tris(vertices, faces, device=device)

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

    e1 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e3 = local_tris[:, 0, :] - local_tris[:, 2, :]
    e1_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 2, :] - paramtris[:, 1, :]
    e3_p = paramtris[:, 0, :] - paramtris[:, 2, :]

    # NOTE: sometimes denominator will be 0 i.e. area of triangle is 0 -> cotangent in this case is infty, default to 1e5
    # Cotangent = cos/sin = dot(adjacent edges)/sqrt(1 - cos^2)
    cos1 = torch.nn.functional.cosine_similarity(-e2, e3)
    cos2 = torch.nn.functional.cosine_similarity(e1, -e3)
    cos3 = torch.nn.functional.cosine_similarity(-e1, e2)

    cot1 = cos1/torch.sqrt(1 - cos1**2)
    cot2 = cos2/torch.sqrt(1 - cos2**2)
    cot3 = cos3/torch.sqrt(1 - cos3**2)

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Threshold param tris as well
    e1_p = torch.maximum(torch.minimum(e1_p, torch.tensor(1e5)), torch.tensor(1e-5))
    e2_p = torch.maximum(torch.minimum(e2_p, torch.tensor(1e5)), torch.tensor(1e-5))
    e3_p = torch.maximum(torch.minimum(e3_p, torch.tensor(1e5)), torch.tensor(1e-5))

    # Compute all edge rotations
    cot_full = torch.stack([cot1, cot2, cot3]).reshape(3, len(cot1), 1, 1)
    e_full = torch.stack([e1, e2, e3])
    e_p_full = torch.stack([e1_p, e2_p, e3_p])

    # Compute covariance matrix
    crosscov = torch.sum(cot_full * torch.matmul(e_p_full.unsqueeze(3), e_full.unsqueeze(2)), dim=0)
    crosscov = crosscov.reshape(crosscov.shape[0], 4) # F x 4

    E = (crosscov[:,0] + crosscov[:,3])/2
    F = (crosscov[:,0] - crosscov[:,3])/2
    G = (crosscov[:,2] + crosscov[:,1])/2
    H = (crosscov[:,2] - crosscov[:,1])/2

    Q = torch.sqrt(E ** 2 + H ** 2)
    R = torch.sqrt(F ** 2 + G ** 2)

    S1 = Q + R
    S2 = Q - R
    a1 = torch.atan2(G, F)
    a2 = torch.atan2(H, E)
    theta = (a2 - a1) / 2 # F
    phi = (a2 + a1) / 2 # F

    # F x 2 x 2
    # NOTE: This is U^T
    U = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1), torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=2)

    # F x 2 x 2
    # NOTE: This is V
    V = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1), torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)], dim=2)

    R = torch.matmul(V, U).to(device) # F x 2 x 2

    ## NOTE: Sanity check the SVD
    S = torch.stack([torch.diag(torch.tensor([S1[i], S2[i]])) for i in range(len(S1))]).to(S1.device)
    checkcov = U.transpose(2,1) @ S @ V.transpose(2,1)
    torch.testing.assert_close(crosscov.reshape(-1, 2, 2), checkcov)

    # Sometimes rotation is opposite orientation: just check with determinant and flip
    # NOTE: Can flip sign of det by flipping sign of last column of V
    baddet = torch.where(torch.det(R) <= 0)[0]
    if len(baddet) > 0:
        print(f"ARAP warning: found {len(baddet)} flipped rotations.")
        V[baddet, :, 1] *= -1
        R = torch.matmul(V, U).to(device) # F x 2 x 2
        assert torch.all(torch.det(R) >= 0)

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
    # from meshing import Mesh
    # from meshing.analysis import computeFaceAreas
    # mesh = Mesh(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())
    # computeFaceAreas(mesh)
    # fareas = torch.from_numpy(mesh.fareas).to(vertices.device)

    # NOTE: We normalize by mean edge length b/w p and e b/c for shrinking ARAP is bounded by edge length
    # Normalizing by avg edge length b/w p and e => ARAP bounded by 2 on BOTH SIDES
    mean_elen = (torch.linalg.norm(e_full, dim=2) + torch.linalg.norm(e_p_full, dim=2))/2
    arap_tris = torch.sum(torch.abs(cot_full) * 1/mean_elen * torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2, dim=0) # F x 1
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

# ==================== Stitching Energies ===============================
def stitchingloss(vs, fs, uv, losstypes, args, stitchweights=None, source = None, keepidxs = None):
    from source_njf.utils import vertex_soup_correspondences
    from itertools import combinations

    uvpairs = uv[source.edge_vpairs.to(uv.device)] # E x 2 x 2 x 2
    elens = source.elens_nobound.to(uv.device)

    if keepidxs is not None:
        uvpairs = uvpairs[keepidxs]
        elens = elens[keepidxs]


    ## Edge separation loss
    if args.stitchdist == 'l2':
        edgesep = torch.sqrt(torch.sum(torch.nn.functional.mse_loss(uvpairs[:,:,0,:], uvpairs[:,:,1,:], reduction='none'), dim=2))
    elif args.stitchdist == 'l1':
        edgesep = torch.sum(torch.nn.functional.l1_loss(uvpairs[:,:,0,:], uvpairs[:,:,1,:], reduction='none'), dim=2)
    edgesep = torch.mean(edgesep, dim=1) # E x 1

    # Weight with edge lengths
    wedgesep = edgesep * elens

    # if stitchweights is None:
    #     stitchweights = torch.ones(len(vertexsep), device=uv.device)

    lossdict = {}
    weightdict = {}
    # NOTE: We assume everything can just use edges going forward ... uncomment if not true
    for losstype in losstypes:
        # if losstype == "vertexseploss": # Pairs x 2
        #     if args.seamlessvertexsep:
        #         lossdict[losstype] = stitchweights * torch.sum(seamlessvertexsep, dim=1)
        #     else:
        #         lossdict[losstype] = stitchweights * torch.sum(vertexsep, dim=1)
        #         weightdict[losstype] = args.vertexsep_weight

        if losstype == "edgecutloss": # E x 1
            if args.seamlessedgecut:
                edgecutloss = wedgesep * wedgesep/(wedgesep * wedgesep + args.seamlessdelta)
            else:
                edgecutloss = wedgesep
            lossdict[losstype] = edgecutloss
            weightdict[losstype] = args.edgecut_weight

        # elif losstype == "edgegradloss": # E x 2
        #     gradloss = uvgradloss(vs, fs, uv, loss=args.stitchdist)

        #     if args.seamlessgradloss:
        #         gradloss = (gradloss * gradloss)/(gradloss * gradloss + args.seamlessdelta)

        #     lossdict[losstype] = gradloss
        #     weightdict[losstype] = args.edgegrad_weight

    return edgesep, lossdict, weightdict

# Compute loss based on vertex-vertex distances
def vertexseparation(vs, fs, uv, loss='l1'):
    """ uv: F * 3 x 2
        vs: V x 3 (original topology)
        fs: F x 3 (original topology) """
    from source_njf.utils import vertex_soup_correspondences
    from itertools import combinations
    from meshing import Mesh
    vcorrespondences = vertex_soup_correspondences(fs.detach().cpu().numpy())
    vpairs = []
    for ogv, vlist in sorted(vcorrespondences.items()):
        vpairs.extend(list(combinations(vlist, 2)))
    vpairs = torch.tensor(vpairs, device=uv.device)
    uvpairs = uv[vpairs] # V x 2 x 2

    if loss == "l1":
        separation = torch.nn.functional.l1_loss(uvpairs[:,0], uvpairs[:,1], reduction='none')
    elif loss == 'l2':
        separation = torch.nn.functional.mse_loss(uvpairs[:,0], uvpairs[:,1], reduction='none')

    return separation

def uvgradloss(vs, fs, uv, return_edge_correspondence=False, loss='l2'):
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
    elens = []
    for k, v in sorted(edgecorrespondences.items()):
        # If only one correspondence, then it is a boundary
        if len(v) == 1:
            continue
        e1.append(uvsoup[list(v[0])])
        e2.append(uvsoup[list(v[1])])
        elens.append(np.linalg.norm(vs[k[0]] - vs[k[1]]))

    ef0 = torch.cat(e1) # E*2 x 2
    ef1 = torch.cat(e2) # E*2 x 2
    elens = torch.tensor(elens, device=uv.device).reshape(len(elens), 1)

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
        separation = elens * torch.nn.functional.l1_loss(e0, e1, reduction='none')
    elif loss == 'l2':
        separation = elens * torch.nn.functional.mse_loss(e0, e1, reduction='none')
    elif loss == 'cosine':
        # Cosine similarity
        separation = elens.squeeze() * (1 - torch.nn.functional.cosine_similarity(e0, e1, eps=1e-8))

    if return_edge_correspondence:
        return separation, edgecorrespondences

    return separation

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
