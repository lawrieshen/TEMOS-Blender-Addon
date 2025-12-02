bl_info = {
    "name": "Motion Generator UI",
    "author": "Lawrence Shen",
    "version": (1, 0),
    "blender": (4, 3, 2),
    "description": "UI to trigger motion generation from deep learning models",
    "category": "Animation",
}

from .ui_panel import (
    MotionGeneratorProperties,
    MotionGeneratorPanel,
    ApplyMotionOperator,
    PreviewMotionModalOperator
)

import bpy

classes = (
    MotionGeneratorProperties,
    MotionGeneratorPanel,
    ApplyMotionOperator,
    PreviewMotionModalOperator
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.motion_generator_props = bpy.props.PointerProperty(type=MotionGeneratorProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.motion_generator_props

if __name__ == "__main__":
    import platform
    print(f'platform: {platform.system()}')
    register()
