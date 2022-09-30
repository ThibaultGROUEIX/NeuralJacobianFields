from concurrent.futures import process
from pathlib import Path
import sys
import igl
import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_njf'))

from DeformationEncoder import DeformationEncoder
from Experiment import Experiment
import time
import fire
import trimesh
from training_scripts.script_train_default import DefaultExperiment
from os.path import join
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from easydict import EasyDict
import numpy as np
try:
    import polyscope as ps
    import polyscope.imgui as psim
    IMPORT_PS = True
except Exception:
    IMPORT_PS = False


class PolyScope(object):
    def __init__(self, source_mesh, target_mesh_list, with_WKS=True, save_obj_to_disk=False) -> None:
        # setup our experiment
        self.source_mesh = source_mesh
        self.cpuonly = True
        self.save_obj_to_disk = save_obj_to_disk
        self.target_name = target_mesh_list
        self.exp = DefaultExperiment(cpuonly=self.cpuonly)
        # load a checkpoint (you should train to get one)
        self.args_to_overwrite = EasyDict()
        self.args_to_overwrite.experiment_type = "DEFAULT"

        if with_WKS:
            print("loading network trained with WKS")
            self.exp.load_network("data/checkpoints/morph_any_shape_into_any_other_withWKS.ckpt") 
        else:
            print("loading network trained withOUT WKS")
            self.exp.load_network("data/checkpoints/morph_any_shape_into_any_other_withoutWKS.ckpt") 
            self.args_to_overwrite.no_wks = True

        if not isinstance(target_mesh_list, list):
            target_mesh_list = [target_mesh_list]
        

        self.scale_and_translate()
        self.template = igl.read_triangle_mesh(source_mesh)
        self.mesh = [igl.read_triangle_mesh(target_name) for target_name in  self.target_name]
        self.faces = [mesh[1] for mesh in self.mesh]

    def extract_largest_connected_component(self, mesh):
        import trimesh
        mesh = trimesh.Trimesh(mesh[0], mesh[1], process=False)
        cp = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        cp_max = np.argmax(np.bincount(cp))
        mesh.update_faces(cp==cp_max)
        mesh.remove_unreferenced_vertices()
        return [mesh.vertices, mesh.faces]

    def scale_and_translate(self):
        self.scales = []
        self.translations = []
        self.bb = []
        for indes, target_name in enumerate(self.target_name):
            self.target_name[indes] = f'{target_name[:-4]}_scaled_and_translated.obj'
            if not Path(self.target_name[indes]).is_file() or  self.save_obj_to_disk:
                mesh = igl.read_triangle_mesh(target_name)
                mesh = self.extract_largest_connected_component(mesh)
                mesh_typical = igl.read_triangle_mesh("./data/source.obj")

                area_typical = trimesh.Trimesh(mesh_typical[0], mesh_typical[1]).area
                area = trimesh.Trimesh(mesh[0], mesh[1]).area
                self.scales.append(area)

                # vert = mesh[0]*igl.bounding_box_diagonal(mesh_typical[0])/igl.bounding_box_diagonal(mesh[0])
                vert = mesh[0]*np.sqrt(area_typical/area)

                bb_typical = igl.bounding_box(mesh_typical[0])[0]
                bb = igl.bounding_box(vert)[0]
                self.bb.append(bb)
                self.translations.append(mesh[0].mean(0, keepdims=True))
                vert -= mesh[0].mean(0, keepdims=True)
                # vert = vert - torch.from_numpy(vert).mean(0).numpy()
                igl.write_obj(self.target_name[indes], vert, mesh[1] )

    def run(self):
        if IMPORT_PS:
            ps.init()
            ps.set_always_redraw(False)
            ps.look_at((0,0,5), (0,0,0))
            ps.set_build_gui(False)
            ps.set_view_projection_mode("orthographic")


    def forward(self):
        with torch.no_grad():
            batch_of_maps_list = []
            source_list = []
            target_list = []

            batch_of_maps, batch_of_jacobians, source, target = self.exp.evaluate_on_source_and_targets_inference(self.source_and_targets[0],self.source_and_targets[1], cpuonly=self.cpuonly)
            batch_of_maps_list.append(batch_of_maps.cpu().numpy()[0])
            source_list.append(source)
            target_list.append(target)
            return batch_of_maps_list, source_list, target_list

    def infer(self):
        for i in range(len(self.target_name)):
            if IMPORT_PS:
                ps.remove_all_structures()
                self.run()
                # source
                
                template = ps.register_surface_mesh(f"mesh_{i}_source",self.template[0], self.template[1], color=(0,1,0))
                template.translate((-1.5,0,0))

                # target
                source_mesh = ps.register_surface_mesh(f"mesh{i}0",self.mesh[i][0], self.faces[i], color=(0,0,1))
                source_mesh.translate((0.5,0,0))
                # target
                source_mesh = ps.register_surface_mesh(f"mesh{i}0f",self.mesh[i][0], self.faces[i], color=(0,0,1))
                source_mesh.translate((-0.5,0,0))

            # source deformed
            self.source_and_targets = self.exp.evaluate_on_source_and_targets_preprocess(self.source_mesh, [self.target_name[i]], cpuonly=self.cpuonly, args_to_overwrite=self.args_to_overwrite)
            batch_of_maps_list, _, _ = self.forward()

            bb = igl.bounding_box(batch_of_maps_list[0])[0]
            bb_target = igl.bounding_box(self.mesh[i][0])[0]
                # (bb_target[0] + self.bb[i][-1])/2 - (bb[0] + bb[-1])/2 
            batch_of_maps_list[0] = batch_of_maps_list[0] + (bb_target[0] + bb_target[-1])/2 - (bb[0] + bb[-1])/2 

            if IMPORT_PS:
                target_ps = ps.register_surface_mesh(f"mesh{i}1",batch_of_maps_list[0], self.template[1], color=(0,1,0))
                # target_ps.translate((1,0,0))
                target_ps_2 = ps.register_surface_mesh(f"mesh{i}2",batch_of_maps_list[0], self.template[1], color=(0,1,0))
                target_ps.translate((1.5,0,0))
                target_ps_2.translate((0.5,0,0))

            name_target = Path(self.target_name[i]).parts[-1][:-26]
            name_source = Path(self.source_mesh).parts[-1][:-4]
            result_dir = Path("./results/morph_any_shape_into_any_other/")
            os.makedirs(result_dir/"renderings", exist_ok=True)
            os.makedirs(result_dir/"objs", exist_ok=True)

            if IMPORT_PS:
                ps.screenshot((result_dir/f"renderings/{name_source}_{name_target}.png").as_posix())

            if self.save_obj_to_disk:
                igl.write_obj((result_dir/"./objs"/f"{name_source}_{name_target}.obj").as_posix(), batch_of_maps_list[0] , self.template[1])

                scale = np.sqrt(self.scales[i] / trimesh.Trimesh(batch_of_maps_list[0], self.template[1]).area) 
                verts = batch_of_maps_list[0] * scale

                bb = igl.bounding_box(verts)[0]
                # (self.bb[i][0] + self.bb[i][-1])/2 - (bb[0] + bb[-1])/2 
                verts = verts + (self.bb[i][0] + self.bb[i][-1])/2 - (bb[0] + bb[-1])/2 
                # verts = verts - verts.mean(0, keepdims=0) + self.translations[i]
                igl.write_obj((Path("./objs")/f"{name_source}_{name_target}.obj").as_posix(), verts, self.template[1])
            print("-- Results generated in ./results/  --")

def main(source_mesh : str = "./data/source.obj",  target_mesh : str = "./data/target.obj", with_WKS=True, save_obj_to_disk=False):
   if Path(target_mesh).is_file():
       target_mesh = [target_mesh]
   elif Path(target_mesh).is_dir():
       import glob
       a = glob.glob(f"{target_mesh}*obj")
       target_mesh = [source for source in a if source[-10:] != "slated.obj"]
    
   for i, target in enumerate(target_mesh):
        if target[-4:] == ".ply":
            mesh = trimesh.load_mesh(target, process=False)
            mesh.export(target[:-4] + ".obj")
            target_mesh[i] = target[:-4] + ".obj"
            
   test = PolyScope(source_mesh, target_mesh, with_WKS=with_WKS, save_obj_to_disk=save_obj_to_disk)
   test.run()
   test.infer()


if __name__ == '__main__':
    # Usage on single mesh : 
        # python demo_scripts/demo_morph.py -source_mesh bunny.obj -target_mesh ./data/00001.obj
    # Usage on folder : 
        # python demo_scripts/demo_morph.py -source_mesh bunny.obj -target_mesh ./data/  
    fire.Fire(main)
