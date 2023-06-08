# Pyvista plotting class
class PVPlotter:
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True)
    pv.rcParams['transparent_background'] = True

    # Plot texture while automatically setting camera view settings based on input mesh and UVs
    def uv_autoplot(self, mesh, anchor, submesh, subanchor, selection, uvmap, texturefile, savename, savedir,
                    plot_uv=False, plot_full=False, plot_soft=False, plot_hard=False, plot_error=False,
                    error=None, error_clim=None, softweights=False, plot_gif=False, log_scale=False,
                    uv_full=None, error_full=None):
        import numpy as np
        import os

        vertices, faces, _ = mesh.export_soup()
        sub_vertices, sub_faces, _ = submesh.export_soup()

        # Set camera parameters
        if not hasattr(mesh, "facenormals"):
            from meshing.analysis import computeFaceNormals
            computeFaceNormals(mesh)

        if not hasattr(submesh, "facenormals"):
            from meshing.analysis import computeFaceNormals
            computeFaceNormals(submesh)

        # Normalize UV map to fall within 0,1
        normuv = uvmap - np.min(uvmap, axis=0, keepdims=True)
        normuv /= np.max(normuv)
        normuv += (0.5 - np.mean(normuv, axis=0))

        if plot_uv == True:
            # Plot 2D embedding
            normvs = np.concatenate([normuv, np.zeros((len(normuv), 1))], axis=1)
            lookat = np.array([0.5, 0.5, 0.0])
            cpos = np.array([0.5, 0.5, 2.5])
            self.plottexture(normvs, sub_faces, normuv, save_path = os.path.join(savedir, savename),
                        texturefile = texturefile, name=savename, cpos=cpos, clook=lookat,
                        anchor=subanchor)

            if plot_error and error is not None:
                # Define scalar bar parameters
                sargs = dict(fmt="%.2g", color='black', n_labels=4)
                self.plottexture(normvs, sub_faces, normuv, save_path = os.path.join(savedir, savename),
                        texturefile = None, name=f"{savename}_error", cpos=cpos, clook=lookat,
                        anchor=subanchor, scalars=error, show_edges=True, sargs=sargs, clim=error_clim,
                        log_scale=log_scale)

        if plot_full == True:
            # Try to visualize patch over original shape
            lookat = np.mean(vertices[faces[anchor].flatten()], axis=0)
            viewvec = mesh.facenormals[anchor].squeeze()
            viewvec /= np.linalg.norm(viewvec)
            # Reverse sign if same direction as mesh centroid
            centroid = np.mean(vertices, axis=0)
            if np.dot(viewvec, centroid - lookat) > 0:
                viewvec *= -1

            # Adjust cpos radius based on viewbox centered at anchor
            ## Full
            projected_vs = vertices - np.dot(vertices - lookat, viewvec).reshape(len(vertices),1) * np.stack([viewvec] * len(vertices))
            project_dist = np.linalg.norm(projected_vs - lookat, axis=1)
            full = np.max(project_dist) * 3.8
            cpos_full = viewvec * full + lookat

            ## Seg
            projected_vs = sub_vertices - np.dot(sub_vertices - lookat, viewvec).reshape(len(sub_vertices),1) * np.stack([viewvec] * len(sub_vertices))
            project_dist = np.linalg.norm(projected_vs - lookat, axis=1)
            seg = np.max(project_dist) * 3.8
            cpos_seg = viewvec * seg + lookat

            # Debugging: check ABOVE PROJECTIONS + THE FINAL CPOS
            # import polyscope as ps
            # ps.init()
            # ps_mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=1)
            # anchorcolors = np.zeros(len(faces))
            # anchorcolors[anchor] = 1
            # ps_mesh.add_scalar_quantity("anchor", anchorcolors, defined_on='faces', enabled=True)
            # selectcolors = np.zeros(len(faces))
            # selectcolors[selection] = 1
            # ps_mesh.add_scalar_quantity("selection", selectcolors, defined_on='faces', enabled=True)
            # ps_project = ps.register_surface_mesh("proj select", projected_vs, sub_faces, edge_width=1)
            # ps.look_at(cpos_full, lookat)
            # ps.show()
            # ps.look_at(cpos_seg, lookat)
            # ps.show()
            # raise

            # Set axes to orthonormal vectors to the anchor normal
            angles = np.linspace(0, 2 * np.pi, 8)
            ortho = np.array([0., 1., 0.])
            up = np.array([0., 1., 0.])
            # Viewvector is close to negative y-axis
            if np.abs(np.dot(viewvec, -ortho)) >= 0.999 or np.abs(np.dot(viewvec, ortho)) >= 0.999:
                ortho = np.array([1., 0., 0.])
                up = np.array([1., 0., 0.])
            axis1 = np.cross(viewvec, ortho)

            # Sometimes axis is norm 0
            if np.linalg.norm(axis1) == 0:
                raise ValueError("Axis 1 has 0 norm.")

            axis1 /= np.linalg.norm(axis1)
            axis2 = np.cross(viewvec, axis1)

            # Sometimes axis is norm 0
            if np.linalg.norm(axis2) == 0:
                raise ValueError("Axis 2 has 0 norm.")

            axis2 /= np.linalg.norm(axis2)

            # Extract other part of mesh
            other_selection = np.array(list(set(range(len(faces))).difference(set(selection))))
            othervs = otherfs = None
            if len(other_selection) > 0:
                othervs, otherfs = mesh.export_submesh(other_selection)

            boundary = None
            if len(submesh.topology.boundaries) > 0:
                boundary = 5
            # boundary = 5

            # TODO: TEMPORARY
            # cpos_full = [-2.4267669, -0.01197113, -3.5192702 ]
            # lookat = [-0.3174843, 0.18658893, -0.3754371 ]
            # axis1 = [ 0.83041386,0.,-0.55714704]
            # axis2 = [ 0.02918101,-0.99862745, 0.04349357]
            # axis1 = [0,1,0]
            # axis2 = [1,0,0]
            # cpos_full = [0, 1.5, 3.5]
            # lookat = [0,0,0]

            # Full render
            self.plottexture(sub_vertices, sub_faces, normuv, save_path = os.path.join(savedir, f"{savename}_full"),
                        texturefile = texturefile, axes = [axis1, axis2], angles = angles, boundary=boundary,
                        name=savename, cpos=cpos_full, clook=lookat, up = up, save_gif=plot_gif, anchor=subanchor,
                        othervs=othervs, otherfs=otherfs, show_edges=True)

            # Seg in viewbox render
            self.plottexture(sub_vertices, sub_faces, normuv, save_path = os.path.join(savedir, f"{savename}_view"),
                        texturefile = texturefile, axes = [axis1, axis2], angles = angles, boundary=boundary,
                        name=savename, cpos=cpos_seg, clook=lookat, up = up, save_gif=plot_gif, anchor=subanchor,
                        othervs=othervs, otherfs=otherfs, show_edges=True)

            # TODO: TEMPORARY
            if uv_full is not None:
                self.plottexture(vertices, faces, uv_full[:,:2], save_path = os.path.join(savedir, f"{savename}_uv_full"),
                            texturefile = texturefile, axes = [axis1, axis2], angles = angles, boundary=boundary,
                            name=savename, cpos=cpos_full, clook=lookat, up = up, save_gif=plot_gif, anchor=anchor)

                # TODO: TEMPORARY
                if plot_error and error_full is not None and uv_full is not None:
                    # Define scalar bar parameters
                    uv_full = uv_full - np.min(uv_full, axis=0, keepdims=True)
                    uv_full /= np.max(uv_full)
                    uv_full = np.concatenate([uv_full, np.zeros((len(uv_full), 1))], axis=1)
                    lookat = np.array([0.5, 0.5, 0.0])
                    cpos = np.array([0.5, 0.5, 2.5])

                    sargs = dict(fmt="%.2g", color='black', n_labels=4)
                    self.plottexture(uv_full, faces, uv_full[:,:2], save_path = os.path.join(savedir, savename),
                            texturefile = None, name=f"{savename}_error_full", cpos=cpos, clook=lookat,
                            anchor=anchor, scalars=error_full, show_edges=True, sargs=sargs, clim=error_clim,
                            log_scale=log_scale)

                    self.plottexture(uv_full, faces, uv_full[:,:2], save_path = os.path.join(savedir, savename),
                            texturefile = texturefile, name=f"{savename}_full", cpos=cpos, clook=lookat,
                            anchor=anchor)

        if plot_soft == True and softweights is not None:
            # Try to visualize patch over original shape
            lookat = np.mean(vertices[faces[anchor].flatten()], axis=0)
            viewvec = mesh.facenormals[anchor].squeeze()
            viewvec /= np.linalg.norm(viewvec)
            # Reverse sign if same direction as mesh centroid
            centroid = np.mean(vertices, axis=0)
            if np.dot(viewvec, centroid - lookat) > 0:
                viewvec *= -1

            # Set two different cpos
            ## Full
            projected_vs = vertices - np.dot(vertices - lookat, viewvec).reshape(len(vertices),1) * np.stack([viewvec] * len(vertices))
            project_dist = np.linalg.norm(projected_vs - lookat, axis=1)
            full = np.max(project_dist) * 3.5
            cpos_full = viewvec * full + lookat

            ## Seg
            projected_vs = sub_vertices - np.dot(sub_vertices - lookat, viewvec).reshape(len(sub_vertices),1) * np.stack([viewvec] * len(sub_vertices))
            project_dist = np.linalg.norm(projected_vs - lookat, axis=1)
            seg = np.max(project_dist) * 3.5
            cpos_seg = viewvec * seg + lookat

            # Set axes to orthonormal vectors to the anchor normal
            angles = np.linspace(0, 2 * np.pi, 8)
            ortho = np.array([0., 1., 0.])
            up = np.array([0., 1., 0.])
            # Viewvector is close to negative y-axis
            if np.abs(np.dot(viewvec, -ortho)) >= 0.999 or np.abs(np.dot(viewvec, ortho)) >= 0.999:
                ortho = np.array([1., 0., 0.])
                up = np.array([1., 0., 0.])
            axis1 = np.cross(viewvec, ortho)

            # Sometimes axis is norm 0
            if np.linalg.norm(axis1) == 0:
                raise ValueError("Axis 1 has 0 norm.")

            axis1 /= np.linalg.norm(axis1)
            axis2 = np.cross(viewvec, axis1)

            # Sometimes axis is norm 0
            if np.linalg.norm(axis2) == 0:
                raise ValueError("Axis 2 has 0 norm.")

            axis2 /= np.linalg.norm(axis2)

            # Extract other part of mesh
            other_selection = np.array(list(set(range(len(faces))).difference(set(selection))))
            othervs = otherfs = None
            if len(other_selection) > 0:
                othervs, otherfs = mesh.export_submesh(other_selection)

            sargs = dict(fmt="%.2f", color='black', n_labels=3)
            self.plottexture(vertices, faces, None, save_path = os.path.join(savedir, f"{savename}_soft_full"),
                        axes = [axis1, axis2], angles = angles, scalars=softweights, clim=[0,1],
                        name=savename, cpos=cpos_full, clook=lookat, up = up, save_gif=plot_gif, anchor=anchor,
                        cmap='viridis', sargs=sargs)

            self.plottexture(vertices, faces, None, save_path = os.path.join(savedir, f"{savename}_soft_view"),
                        axes = [axis1, axis2], angles = angles, scalars=softweights, clim=[0,1],
                        name=savename, cpos=cpos_seg, clook=lookat, up = up, save_gif=plot_gif, anchor=anchor,
                        cmap='viridis', sargs=sargs)

        if plot_hard == True:
            # Try to visualize patch over original shape
            lookat = np.mean(vertices[faces[anchor].flatten()], axis=0)
            viewvec = mesh.facenormals[anchor].squeeze()
            viewvec /= np.linalg.norm(viewvec)
            # Reverse sign if same direction as mesh centroid
            centroid = np.mean(vertices, axis=0)
            if np.dot(viewvec, centroid - lookat) > 0:
                viewvec *= -1

            # Set two different cpos
            ## Full
            projected_vs = vertices - np.dot(vertices - lookat, viewvec).reshape(len(vertices),1) * np.stack([viewvec] * len(vertices))
            project_dist = np.linalg.norm(projected_vs - lookat, axis=1)
            full = np.max(project_dist) * 3.5
            cpos_full = viewvec * full + lookat

            ## Seg
            projected_vs = sub_vertices - np.dot(sub_vertices - lookat, viewvec).reshape(len(sub_vertices),1) * np.stack([viewvec] * len(sub_vertices))
            project_dist = np.linalg.norm(projected_vs - lookat, axis=1)
            seg = np.max(project_dist) * 3.5
            cpos_seg = viewvec * seg + lookat

            # Set axes to orthonormal vectors to the anchor normal
            angles = np.linspace(0, 2 * np.pi, 8)
            ortho = np.array([0., 1., 0.])
            up = np.array([0., 1., 0.])
            # Viewvector is close to negative y-axis
            if np.abs(np.dot(viewvec, -ortho)) >= 0.999 or np.abs(np.dot(viewvec, ortho)) >= 0.999:
                ortho = np.array([1., 0., 0.])
                up = np.array([1., 0., 0.])
            axis1 = np.cross(viewvec, ortho)

            # Sometimes axis is norm 0
            if np.linalg.norm(axis1) == 0:
                raise ValueError("Axis 1 has 0 norm.")

            axis1 /= np.linalg.norm(axis1)
            axis2 = np.cross(viewvec, axis1)

            # Sometimes axis is norm 0
            if np.linalg.norm(axis2) == 0:
                raise ValueError("Axis 2 has 0 norm.")

            axis2 /= np.linalg.norm(axis2)

            # Extract other part of mesh
            other_selection = np.array(list(set(range(len(faces))).difference(set(selection))))
            othervs, otherfs = mesh.export_submesh(other_selection)

            hardweights = np.zeros(len(faces))
            hardweights[selection] = 1

            sargs = dict(fmt="%.2f", color='black', n_labels=3)
            self.plottexture(vertices, faces, None, save_path = os.path.join(savedir, f"{savename}_soft_full"),
                        axes = [axis1, axis2], angles = angles, scalars=hardweights, clim=[0,1],
                        name=savename, cpos=cpos_full, clook=lookat, up = up, save_gif=plot_gif, anchor=anchor,
                        cmap='viridis', sargs=sargs)

            self.plottexture(vertices, faces, None, save_path = os.path.join(savedir, f"{savename}_soft_view"),
                        axes = [axis1, axis2], angles = angles, scalars=hardweights, clim=[0,1],
                        name=savename, cpos=cpos_seg, clook=lookat, up = up, save_gif=plot_gif, anchor=anchor,
                        cmap='viridis', sargs=sargs)

    def plottexture(self, vertices, faces, uv, save_path, texturefile = None, axes = None, angles = None, name = "mesh",
                anchor=None, frame_folder = "frames", othervs=None, otherfs=None, boundary=None, sargs=None,
                cpos = None, clook=None, up=[0,1,0], save_gif=False, show_edges=False, scalars=None, cmap='cividis_r', clim=None,
                log_scale = False, othercolor=None):
        from PIL import Image
        import pyvista as pv
        from utils import getRotMat
        import os
        from pathlib import Path
        import numpy as np

        self.plotter.clear()

        # Pyvista needs faces data to include # vertex count
        if faces.shape[1] < 4:
            faces = np.concatenate([np.ones((len(faces), 1)) * 3, faces], axis=1).astype(int)
        if otherfs is not None:
            if otherfs.shape[1] < 4:
                otherfs = np.concatenate([np.ones((len(otherfs), 1)) * 3, otherfs], axis=1).astype(int)

        frame_path = os.path.join(save_path, frame_folder)
        Path(frame_path).mkdir(parents=True, exist_ok=True)

        # Set background to transparent by default
        self.plotter.camera_set = True

        tex = None
        if texturefile:
            texture = Image.open(texturefile)
            teximage = np.array(texture)
            texture.close()
            tex = pv.numpy_to_texture(teximage)

        count = 0
        if axes is None or angles is None:
            pvmesh = pv.PolyData(vertices, faces)
            if uv is not None:
                pvmesh.active_t_coords = uv
            cparams = None
            if cpos is not None and clook is not None:
                cparams = [cpos, clook, up]
            self.plotter.camera.position = cpos
            self.plotter.camera.focal_point = clook
            self.plotter.camera.up = up

            pvactor = self.plotter.add_mesh(pvmesh, texture=tex, show_edges=show_edges, scalars=scalars, cmap=cmap, clim=clim,
                                    scalar_bar_args=sargs, log_scale = log_scale)
            if clim is not None:
                self.plotter.remove_scalar_bar()

            if othervs is not None and otherfs is not None:
                otheractor = self.plotter.add_mesh(pv.PolyData(othervs, otherfs), show_edges=show_edges, color=othercolor,)
            if anchor:
                anchorpos = np.mean(vertices[faces[anchor,1:].flatten()], axis=0)
                fverts = vertices[faces[anchor,1:].flatten()]
                sidelengths = np.linalg.norm(np.stack([fverts[0] - fverts[1], fverts[0] - fverts[2], fverts[1] - fverts[2]]), axis=1)
                s = np.sum(sidelengths)/2
                area = np.sqrt(s * (s - sidelengths[0]) * (s - sidelengths[1]) * (s - sidelengths[2]))
                anchorrad =  np.prod(sidelengths)/(4 * area) # Circumradius of triangle
                anchorrad = np.min(sidelengths) # Set radius to min edge length instead
                sphere = pv.Sphere(radius=anchorrad/4, center=anchorpos)
                pvsphere = self.plotter.add_mesh(sphere, color=[255,0,0])

            bdry = None
            if boundary:
                bdry = pvmesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
                pvbdry = self.plotter.add_mesh(bdry, color="#FFD700", line_width=boundary)

            # plotter.show(screenshot = os.path.join(save_path, f"{name}.png"))
            self.plotter.screenshot(os.path.join(save_path, f"{name}.png"), transparent_background=True)

            self.plotter.remove_actor(pvactor)
            if anchor:
                self.plotter.remove_actor(pvsphere)
            if othervs is not None and otherfs is not None:
                self.plotter.remove_actor(otheractor)
            if bdry:
                self.plotter.remove_actor(pvbdry)
            self.plotter.clear()
        else:
            for axis in axes:
                for i in range(len(angles)):
                    rot = getRotMat(axis, angles[i])
                    rot_verts = np.transpose(rot @ np.transpose(vertices))
                    pvmesh = pv.PolyData(rot_verts, faces)
                    if uv is not None:
                        pvmesh.active_t_coords = uv
                    cparams = None
                    if cpos is not None and clook is not None:
                        cparams = [cpos, clook, up]
                    self.plotter.camera.position = cpos
                    self.plotter.camera.focal_point = clook
                    self.plotter.camera.up = up

                    pvactor = self.plotter.add_mesh(pvmesh, texture=tex, show_edges=show_edges, scalars=scalars, cmap=cmap, clim=clim,
                                            scalar_bar_args=sargs, log_scale=log_scale)
                    if clim is not None:
                        self.plotter.remove_scalar_bar()

                    if othervs is not None and otherfs is not None:
                        otherrot_vs = np.transpose(rot @ np.transpose(othervs))
                        othermesh = pv.PolyData(otherrot_vs, otherfs)
                        otheractor = self.plotter.add_mesh(othermesh, color=othercolor, show_edges=show_edges)

                    if anchor:
                        anchorpos = np.mean(rot_verts[faces[anchor,1:].flatten()], axis=0)
                        fverts = vertices[faces[anchor,1:].flatten()]
                        sidelengths = np.linalg.norm(np.stack([fverts[0] - fverts[1], fverts[0] - fverts[2], fverts[1] - fverts[2]]), axis=1)
                        s = np.sum(sidelengths)/2
                        area = np.sqrt(s * (s - sidelengths[0]) * (s - sidelengths[1]) * (s - sidelengths[2]))
                        anchorrad =  np.prod(sidelengths)/(4 * area) # Circumradius of triangle
                        anchorrad = np.min(sidelengths) # Set radius to min edge length instead

                        sphere = pv.Sphere(radius=anchorrad/3, center=anchorpos)
                        pvsphere = self.plotter.add_mesh(sphere, color=[255, 0, 0])

                    bdry = None
                    if boundary:
                        bdry = pvmesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
                        pvbdry = self.plotter.add_mesh(bdry, color="red", line_width=boundary)

                    # plotter.show(screenshot = os.path.join(frame_path, f"{name}_{count:03}.png"))
                    self.plotter.screenshot(os.path.join(frame_path, f"{name}_{count:03}.png"), transparent_background=True)

                    self.plotter.remove_actor(pvactor)
                    if anchor:
                        self.plotter.remove_actor(pvsphere)
                    if othervs is not None and otherfs is not None:
                        self.plotter.remove_actor(otheractor)
                    if bdry and len(bdry.faces) > 0:
                        self.plotter.remove_actor(pvbdry)
                    self.plotter.clear()

                    count += 1

            if save_gif == True:
                import glob
                from PIL import Image
                fp_in = f"{frame_path}/{name}_*.png"
                fp_out = f"{save_path}/{name}.gif"
                img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
                img.save(fp=fp_out, format='GIF', append_images=imgs,
                        save_all=True, duration=200, loop=0, disposal=2)

                # Close everything
                img.close()
                for tmp in imgs:
                    tmp.close()