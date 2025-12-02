import subprocess
import os
import bpy
import platform
import sys
import numpy as np 
from omegaconf import DictConfig 
from pathlib import Path

MOTION_GENERATOR_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = os.path.dirname(MOTION_GENERATOR_PATH)
TEMOS_PATH = os.path.join(PROJECT_PATH, "models", "TEMOS")

TEMOS_INTERACT_SCRIPT = os.path.join(TEMOS_PATH, "interact.py")
TEMOS_RENDER_SCRIPT = os.path.join(TEMOS_PATH, "render.py")
PRETRAINED_TEMOS_MODEL_PATH = os.path.join(TEMOS_PATH, "pretrained_models", "kit-amass-rot", "1cp6dwpa")

NPY_PNG_OUTPUT_DIR = os.path.join(TEMOS_PATH, "tmp", "npy_png_output")
RIG_NPZ_PATH = os.path.join(TEMOS_PATH, "tmp", "rig_npz_output")

if platform.system() == "Darwin":
    BLENDER_EXECUTABLE = "/Applications/Blender.app/Contents/MacOS/Blender"
elif platform.system() == "Windows":
    BLENDER_EXECUTABLE = "blender.exe"
else:
    BLENDER_EXECUTABLE = "blender"

def to_cmd_string(path):
    return path

if TEMOS_PATH not in sys.path:
    sys.path.append(TEMOS_PATH)

class AnimationConfig:
    """Mimics the Hydra DictConfig for direct function calls."""
    def __init__(self, npy_path, rig_npz_path):
        self.npy = npy_path
        self.rig_npz_path = rig_npz_path
        
        self.mode = "animation"
        
        self.folder = None
        self.split = "gtest"
        self.infolder = True
        self.jointstype = "vertices"
        self.number_of_samples = 1
        self.denoising = True
        self.oldrender = True
        self.res = "high"
        self.canonicalize = True
        self.exact_frame = 0.5
        self.num = 8
        self.faces_path = os.path.join(TEMOS_PATH, "deps", "smplh", "smplh.faces")
        self.downsample = True
        self.always_on_floor = False
        self.init = True
        self.gt = False
        self.vid_ext = "webm"

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)

try:
    import temos.render.blender
    
    from animate import animate_cli 

except Exception as e:
    print(f"ERROR: Cannot import TEMOS animation core logic. Details: {e}")
    def animate_cli(cfg):
        print("TEMOS animation core is unavailable. Motion cannot be applied to the viewport.")

def _ensure_output_dir():
    """Ensures output directories exist."""
    os.makedirs(NPY_PNG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RIG_NPZ_PATH, exist_ok=True)

def _load_preview_image(image_path):
    """Loads a preview image into Blender's data system."""
    try:
        for img in bpy.data.images:
            if bpy.path.abspath(img.filepath) == image_path:
                return img.name
        image = bpy.data.images.load(image_path)
        return image.name
    except Exception as e:
        print(f"Error loading preview image: {e}")
        raise RuntimeError(f"Failed to load image for preview: {e}")

def run_temos(prompt, length):
    """Runs the external TEMOS model for motion generation and preview rendering."""
    _ensure_output_dir()
    
    motion_generation_cmd = [
        "conda", "run", "-n", "temos", "python", TEMOS_INTERACT_SCRIPT,
        f'folder={PRETRAINED_TEMOS_MODEL_PATH}',
        f'jointstype=vertices',
        f'text={prompt}',
        f'saving={NPY_PNG_OUTPUT_DIR}',
        f'length={length}'
    ]

    print(f"[controller] Running TEMOS: {' '.join(motion_generation_cmd)}")
    subprocess.run(motion_generation_cmd, check=True, cwd=TEMOS_PATH)

    filename = (prompt
                .lower()
                .strip()
                .replace(" ", "_")
                .replace(".", "") + "_len_" + str(length)
                )
    
    generated_npy_path = os.path.join(NPY_PNG_OUTPUT_DIR, f"{filename}.npy")
    preview_png_path = os.path.join(NPY_PNG_OUTPUT_DIR, f"{filename}.png")
    rig_npz_path = os.path.join(RIG_NPZ_PATH, f"{filename}_rig.npz")


    render_sequence_cmd = [
        BLENDER_EXECUTABLE, "--background", "--python", TEMOS_RENDER_SCRIPT,
        "--",
        f'npy={generated_npy_path}',
        f'mode=sequence'
    ]

    print(f"[controller] Rendering Preview: {' '.join(render_sequence_cmd)}")
    subprocess.run(render_sequence_cmd, check=True)

    if not os.path.exists(preview_png_path):
        raise FileNotFoundError(f"Preview PNG not found at {preview_png_path}")
    if not os.path.exists(rig_npz_path):
        raise FileNotFoundError(f"Rig NPZ not found at {rig_npz_path}")

    image_name = _load_preview_image(preview_png_path)
    
    return preview_png_path, image_name, rig_npz_path


def animate_in_viewport_temos(npy_path, rig_npz_path):
    """Applies motion to the currently open Blender scene using TEMOS in-process logic."""
    print(f"[controller] Applying motion in CURRENT viewport: {npy_path}")
    
    cfg = AnimationConfig(npy_path, rig_npz_path)

    animate_cli(cfg)

    bpy.context.view_layer.update()

    print("[controller] Motion application finished.")

def run_animate_pipeline(model, npy_path, rig_npz_path):
    if model == "TEMOS":
        animate_in_viewport_temos(npy_path, rig_npz_path)
    else:
        raise ValueError(f"Unknown model: {model}")
    
def run_preview_pipeline(prompt, model, length):
    if model == "TEMOS":
        return run_temos(prompt=prompt, length=length)
    else:
        raise ValueError(f"Unknown model: {model}")