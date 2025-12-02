import bpy
from . import controller
import os

class MotionGeneratorProperties(bpy.types.PropertyGroup):
    prompt: bpy.props.StringProperty(
        name="Prompt",
        description="Describe the motion",
        default="A person walks and waves"
    )
    length: bpy.props.IntProperty(
        name="Length (seconds)",
        description="Length of the generated motion in seconds",
        default=2,
        min=1,
        max=6
    )
    selected_model: bpy.props.EnumProperty(
        name="Model",
        items=[
            ('TEMOS', "TEMOS", "Generate motion using TEMOS"),
        ],
        default='TEMOS'
    )
    preview_ready: bpy.props.BoolProperty(
        name="Preview Ready",
        default=False
    )
    npy_path: bpy.props.StringProperty(
        name="Vertices npy file Path",
        default=""
    )

    rig_npz_path: bpy.props.StringProperty(
        name="Rig npz file path",
        default=""
    )

    preview_image_name: bpy.props.StringProperty(
        name="Preview Image Name",
        default=""
    )

    status: bpy.props.StringProperty(
        name="Status",
        default="Ready"
    )

    preview_image: bpy.props.PointerProperty(
        name="Preview Image",
        type=bpy.types.Image
    )

    is_previewing: bpy.props.BoolProperty(
        name="Is Previewing",
        default=False
    )
    
class PreviewMotionModalOperator(bpy.types.Operator):
    bl_idname = "wm.preview_motion_modal"
    bl_label = "Preview Motion"
    
    _timer = None
    _started = False

    def modal(self, context, event):
        props = context.scene.motion_generator_props

        if event.type == 'ESC':
            self.report({'INFO'}, "Preview cancelled")
            props.is_previewing = False
            context.window_manager.event_timer_remove(self._timer)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if not self._started:
                self._started = True
                try:
                    npy_path, image_name, rig_npz_path = controller.run_preview_pipeline(
                        prompt=props.prompt,
                        model=props.selected_model,
                        length=int(props.length * 12.5)
                    )
                    props.npy_path = str(npy_path)
                    props.rig_npz_path = str(rig_npz_path)
                    props.preview_image_name = image_name
                    props.preview_image = bpy.data.images.get(image_name)
                    props.preview_ready = True
                    props.status = "Preview ready!"
                except Exception as e:
                    props.status = f"Error: {e}"
                    self.report({'ERROR'}, str(e))
                
                props.is_previewing = False
                context.window_manager.event_timer_remove(self._timer)
                
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        props = context.scene.motion_generator_props

        if not props.selected_model:
            self.report({'WARNING'}, "No model selected.")
            props.is_previewing = False
            return {'CANCELLED'}

        props.status = f"Previewing with {props.selected_model}..."
        props.preview_ready = False
        props.preview_image = None
        props.is_previewing = True

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
class ApplyMotionOperator(bpy.types.Operator):
    bl_idname = "wm.apply_motion"
    bl_label = "Generate Motion"

    def execute(self, context):
        props = context.scene.motion_generator_props

        if not props.preview_ready:
            self.report({'WARNING'}, "No preview generated.")
            return {'CANCELLED'}

        props.status = f"Applying {props.selected_model} motion..."

        try:
            controller.run_animate_pipeline(
                model=props.selected_model,
                npy_path=props.npy_path.replace(".png", ".npy"),  # infer .npy path
                rig_npz_path=props.rig_npz_path
            )
            props.status = "Motion applied to viewport."
        except Exception as e:
            props.status = f"Error: {e}"
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        return {'FINISHED'}


class MotionGeneratorPanel(bpy.types.Panel):
    bl_label = "Motion Generator"
    bl_idname = "VIEW3D_PT_motion_generator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MotionGen'

    def draw(self, context):
        layout = self.layout
        props = context.scene.motion_generator_props

        layout.prop(props, "prompt")
        layout.prop(props, "length")
        layout.prop(props, "selected_model")

        row = layout.row()
        row.enabled = bool(props.selected_model and props.prompt.strip()) and not props.is_previewing
        row.operator("wm.preview_motion_modal", icon='RENDER_RESULT')


        if props.preview_ready and props.preview_image:
            try:
                layout.template_ID_preview(context.scene.motion_generator_props, "preview_image", rows=2, cols=6)
            except Exception as e:
                print(f"Error: {e.with_traceback}")
                layout.label(text=f"Preview load failed: {e}")

        row = layout.row()
        row.enabled = props.preview_ready
        row.operator("wm.apply_motion", icon='CHECKMARK')

        if "Previewing" in props.status:
            layout.label(text=props.status, icon='FILE_REFRESH')  # ⏳ spinner icon
        else:
            layout.label(text=props.status)