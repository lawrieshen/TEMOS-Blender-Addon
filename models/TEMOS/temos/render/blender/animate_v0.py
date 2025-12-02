# temos.render.blender.animate

import bpy
import os
import sys
import numpy as np
import math

from .scene import setup_scene  # noqa
from .floor import show_traj, plot_floor, get_trajectory
from .vertices import prepare_vertices
from .tools import load_numpy_vertices_into_blender, delete_objs, mesh_detect
from .camera import Camera
from .sampler import get_frameidx

from typing import Dict

def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)

def load_smplh_model(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load the SMPLH model data from the given .npz file.

    Args:
        npz_path (str): Path to the SMPLH_NEUTRAL.npz file

    Returns:
        Dict[str, np.ndarray]: A dictionary containing keys:
            - 'v_template' : (6890, 3) vertex positions
            - 'weights'    : (6890, 52) skinning weights
            - 'f'          : (13776, 3) triangle face indices
            - 'kintree_table' : (2, 52) bone hierarchy
            - 'J_regressor': (52, 6890) joint regressor matrix
    """
    data = np.load(npz_path)

    return {
        "v_template": data["v_template"],      # base mesh
        "weights": data["weights"],            # skinning weights
        "f": data["f"],                        # face indices
        "kintree_table": data["kintree_table"],# joint hierarchy
        "J_regressor": data["J_regressor"]     # regresses joints from mesh
    }


def animate_v0(npydata, *, mode, faces_path, gt=False,
           exact_frame=None, num=8, downsample=True,
           canonicalize=True, always_on_floor=False, denoising=True,
           oldrender=True,
           res="high", init=True, rotsdata, transdata):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender)

    is_mesh = mesh_detect(npydata)

    # Data shape of npydata: (25, 6890, 3)
    # (T, V, 3)

    print(f"Data shape of rots data: {rotsdata.shape}")
    print(f"Data shape of trans data: {transdata.shape}")

    if is_mesh:
        from .meshes import Meshes
        data = Meshes(npydata, gt=gt, mode=mode,
                      faces_path=faces_path,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)
    else:
        from .joints import Joints
        data = Joints(npydata, gt=gt, mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)

    # Number of frames possible to render
    nframes = len(data)
    print(f"nframes: {nframes}")

    # Set up the frame rate
    bpy.context.scene.render.fps = 25  # Use higher FPS
    bpy.context.scene.frame_end = int(nframes * 2)

    # Show the trajectory
    show_traj(data.trajectory)

    # Create a floor
    plot_floor(data.data)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    imported_obj_names = []
    shape_keys = []

    # First loop: create and store shape keys
    for frame_index in range(nframes):
        actual_frame = frame_index * 2
        mat = data.mat
        camera.update(data.get_root(frame_index))
        bpy.context.scene.frame_set(actual_frame)

        if frame_index == 0:
            obj, objname = data.load_in_blender_with_return(frame_index, mat)
            imported_obj_names.append(obj)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shape_key_add(from_mix=False)  # Basis
        else:
            obj = bpy.data.objects.get(objname)
            verts = data.data[frame_index]
            key = obj.shape_key_add(name=f"frame_{frame_index}", from_mix=False)
            import mathutils
            for i, v in enumerate(verts):
                key.data[i].co = mathutils.Vector(v)
            shape_keys.append(key)

    # Second loop: insert keyframes per frame
    for frame_index, key in enumerate(shape_keys):
        actual_frame = (frame_index + 1) * 2  # +1 because frame 0 is Basis
        for k in obj.data.shape_keys.key_blocks:
            k.value = 1.0 if k == key else 0.0
            k.keyframe_insert(data_path="value", frame=actual_frame)
            k.keyframe_insert(data_path="value", frame=actual_frame + 1)


