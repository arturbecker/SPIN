import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import constants

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, pickle_path, frame):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0]) #this is why the smpl gt model is being rotated upside down. However, when comment out the next line, it still doesn't quite sit on the right spot, even if we use the uncropped image.
        #mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4) #starting the camera_pose as an identity matrix, so that it already has the fourth row padding with 1 on 3,3 (4th column, 4th row), and a null rotation on 0:2,0:2 (upper left 3x3, which corresponds to the rotation matrix)
        #camera_pose[:3, 3] = camera_translation #this makes sense, the translation is a 3x1 vector on the fourth column of the camera_pose matrix (aka extrinsic parameters)

	# let's instead load the ground truth camera pose parameters 'extrinsic and intrinsic', 'cam_poses' and 'cam_intrinsics', respectively)

        import pickle as pkl
        import os

        seq = pkl.load(open(pickle_path,'rb'),encoding='latin-1')

        camera_translation_gt = seq['cam_poses'][frame][:3, 3]/4.82
        camera_translation_gt[0] *= -1.
        
        camera_pose[:3, 3] = camera_translation_gt
        camera_pose[2, 3] = 2*constants.FOCAL_LENGTH/(constants.IMG_RES * seq['cam_poses'][frame][2,3] +1e-9)
        #res_factor = (seq['cam_intrinsics'][0,2]/(224/2))

        #self.camera_center[0] = seq['cam_intrinsics'][0,2]/res_factor
        #self.camera_center[1] = seq['cam_intrinsics'][1,2]/res_factor

        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[0]) #creating the intrinsics matrix
        #camera = pyrender.IntrinsicsCamera(fx=seq['cam_intrinsics'][0,0], fy=seq['cam_intrinsics'][1,1],
        #                                   cx=seq['cam_intrinsics'][0,2], cy=seq['cam_intrinsics'][1,2])

        scene.add(camera, pose=camera_pose) #putting the camera on the scene


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
