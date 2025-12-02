import numpy as np
from .meshes import Meshes
import bpy
from collections import deque, defaultdict
import mathutils
from mathutils import Matrix, Vector

class ArmatureMotion(Meshes):
    def __init__(self, data, rots, trans, smplh_model, *, gt, mode, faces_path, canonicalize, always_on_floor, oldrender=True, **kwargs):
        """
        data: (T, 6890, 3) baked mesh for compatibility (optional use)
        rots: (T, 22, 3, 3) SMPL joint rotations
        trans: (T, 3) SMPL root translations
        smplh_model: dict from SMPLH_NEUTRAL.npz
        """

        super().__init__(
            data=data, gt=gt, mode=mode, faces_path=faces_path,
            canonicalize=canonicalize, always_on_floor=always_on_floor,
            oldrender=oldrender
        )

        # Check and store rots
        if rots is None or not isinstance(rots, np.ndarray):
            raise ValueError("`rots` must be a NumPy array.")
        if rots.ndim != 4 or rots.shape[2:] != (3, 3):
            raise ValueError(f"`rots` should have shape (T, J, 3, 3), but got {rots.shape}")
        print(rots.shape)

        # Check and store trans
        if trans is None or not isinstance(trans, np.ndarray):
            raise ValueError("`trans` must be a NumPy array.")
        if trans.ndim != 2 or trans.shape[1] != 3:
            raise ValueError(f"`trans` should have shape (T, 3), but got {trans.shape}")

        # Align SMPLH motion to Blender coordinate system
        self.rots, self.trans = self._prepare_smplh_motion(rots, trans)

        # Check and store smpl_model
        required_keys = ["v_template", "weights", "f", "kintree_table", "J_regressor"]
        if smplh_model is None or not isinstance(smplh_model, dict):
            raise ValueError("`smpl_model` must be a dictionary from npz.")
        for key in required_keys:
            if key not in smplh_model:
                raise KeyError(f"Missing required key `{key}` in `smpl_model`.")
            
        self.smplh_model = smplh_model

        self.num_joints = self.rots.shape[1]
        self.num_frames = self.rots.shape[0]

        # Store for convenience
        self.v_template = smplh_model["v_template"]     # (6890, 3)
        self.weights = smplh_model["weights"]           # (6890, 52)
        self.faces = smplh_model["f"]                   # (13776, 3)
        self.kintree_table = smplh_model["kintree_table"]  # (2, 52)
        self.J_regressor = smplh_model["J_regressor"]   # (52, 6890)

        # Others
        self.armature_object = None
        self.mesh_object = None
        self.always_on_floor = always_on_floor
        self.bone_lookup = {} # (parent_joint, child_joint) → bone name

    def load_template_armature(self, armature_name="SMPL_Armature"):
        """Construct armature based on SMPL joint topology."""
        print("Creating armature...")

        # Create new armature and object
        armature_data = bpy.data.armatures.new(armature_name)
        armature_object = bpy.data.objects.new(armature_name, armature_data)
        bpy.context.collection.objects.link(armature_object)

        # Set as active object and enter edit mode
        bpy.context.view_layer.objects.active = armature_object
        bpy.ops.object.mode_set(mode='EDIT')

        # Reconstruct bone hirerachy from knintree
        parents = self.kintree_table[0] # parent array
        children = self.kintree_table[1] # child array

        # Rotate around X-axis by -90 degrees to align SMPL Z-up to Blender Z-up
        R = mathutils.Matrix.Rotation(-np.pi / 2, 4, 'X')

        # compute joint positons from v_template and J_regressor
        J_regressor = self.J_regressor
        v_template = self.v_template

        # joint_positions = J_regressor @ v_template # (52 ,3)
        # Rotation matrix to go from SMPL Z-up to Blender Z-up (rotate -90° around X)
        R_x = np.array([
            [1, 0,  0],
            [0, 0, -1],
            [0, 1,  0]
        ])

        R_z = np.array([
            [0, 1, 0],
            [-1,  0, 0],
            [0,  0, 1]
        ])

        R = R_z @ R_x

        # Rotate all joint positions
        joint_positions = (J_regressor @ v_template) @ R.T  # shape: (52, 3)

        # Flip X and Y for translation
        joint_positions[..., 0] = -joint_positions[..., 0]
        joint_positions[..., 1] = -joint_positions[..., 1]

        if self.always_on_floor:
            joint_positions[..., 2] -= joint_positions[..., 2].min()

        from collections import defaultdict
        child_map = defaultdict(list)
        for parent, child in zip(parents, children):
            if parent != -1 or parent != 4294967295:
                child_map[parent].append(child)
        self

        # Create all bones with head and dummy tail
        num_joints = joint_positions.shape[0]

        # Explicitly handle Bone_0
        bone = armature_data.edit_bones.new("Bone_0")
        bone.head = mathutils.Vector(joint_positions[0]) + mathutils.Vector((-0.01, 0, 0))
        bone.tail = mathutils.Vector(joint_positions[0])
        self.bone_lookup[(-1, 0)] = "Bone_0"
        
        for i in range(num_joints):
            joint = joint_positions[i]
            children = child_map.get(i, [])

            if len(children) == 1:
                # Single child → use default naming
                child = children[0]
                bone_name = f"Bone_{child}"
                bone = armature_data.edit_bones.new(bone_name)
                bone.head = mathutils.Vector(joint)
                bone.tail = mathutils.Vector(joint_positions[child])
                self.bone_lookup[(i, child)] = bone.name

            elif len(children) > 1:
                for j, child in enumerate(children):
                    bone_name = f"Bone_{child}"
                    bone = armature_data.edit_bones.new(bone_name)
                    bone.head = mathutils.Vector(joint)
                    bone.tail = mathutils.Vector(joint_positions[child])
                    self.bone_lookup[(i, child)] = bone.name
            else:
                # Terminal joint → create dummy tail
                bone.tail = bone.head + mathutils.Vector((0, 0, 0.01))
                # continue

        # Set parent relationships
        for (parent_joint, child_joint), child_bone_name in self.bone_lookup.items():
            # Find if this child is also a parent of another joint
            for grandchild in child_map.get(child_joint, []):
                # Look for bone from this child_joint → grandchild
                next_bone_name = self.bone_lookup.get((child_joint, grandchild))
                if next_bone_name:
                    # Set parent relationship
                    if child_bone_name in armature_data.edit_bones and next_bone_name in armature_data.edit_bones:
                        armature_data.edit_bones[next_bone_name].parent = armature_data.edit_bones[child_bone_name]
                        armature_data.edit_bones[next_bone_name].use_connect = True

        
        self.armature_object = armature_object
        print(f"Armature created with {len(armature_data.edit_bones)} bones.")
        bpy.ops.object.mode_set(mode='OBJECT')

    def load_template_mesh(self, name="SMPL_Template", material=None):
        """
        Load the SMPL template mesh (v_template) into the Blender viewport.

        Args:
            name (str): Name for the mesh object.
            material (bpy.types.Material, optional): Material to apply.
        """
        print("Loading SMPL template mesh...")

        vertices = self.v_template
        faces = self.faces

        # Apply the same coordinate transform as joints
        R_x = np.array([
            [1, 0,  0],
            [0, 0, -1],
            [0, 1,  0]
        ])
        R_z = np.array([
            [0, 1, 0],
            [-1,  0, 0],
            [0,  0, 1]
        ])
        R = R_z @ R_x
        vertices = vertices @ R.T

        # Flip X and Y to match Blender
        vertices[:, 0] = -vertices[:, 0]
        vertices[:, 1] = -vertices[:, 1]

        # Align mesh to floor if required
        if self.always_on_floor:
            vertices[:, 2] -= vertices[:, 2].min()

        # Create a new mesh and object
        mesh_data = bpy.data.meshes.new(name + "_Mesh")
        mesh_obj = bpy.data.objects.new(name, mesh_data)

        bpy.context.collection.objects.link(mesh_obj)

        # Create the mesh
        mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh_data.update()

        # Apply material if given
        if material is not None:
            if len(mesh_obj.data.materials) == 0:
                mesh_obj.data.materials.append(material)
            else:
                mesh_obj.data.materials[0] = material

        # Parent to armature if available
        # if self.armature_object:
        #     mesh_obj.parent = self.armature_object

        print("Template mesh loaded.")
        self.mesh_object = mesh_obj

    def apply_motion_to_armature(self):
        """
        Applies SMPL motion (self.rots and self.trans) to the pose bones
        of the already loaded armature across the first frame.
        """
        import bpy
        import numpy as np
        from mathutils import Matrix, Vector

        if self.armature_object is None:
            raise RuntimeError("Armature not created. Call load_template_armature() first.")

        bpy.context.view_layer.objects.active = self.armature_object
        bpy.ops.object.mode_set(mode='POSE')

        pose_bones = self.armature_object.pose.bones
        parents = self.kintree_table[0]
        num_joints = 22  # or whatever number you're visualizing

        # SMPL → Blender coordinate rotation
        R_z = np.array([
            [0, 1, 0], 
            [-1, 0, 0], 
            [0, 0, 1]]
        )
        R_x = np.array([
            [1, 0, 0], 
            [0, 0, -1], 
            [0, 1, 0]]
        )
        R_y = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0]
        ])
        R_smpl2blender = R_y @ R_z @ R_x


    def visualize_smpl_motion(self):
        """
        Visualize the motion encoded in `self.rots` and `self.trans` by computing
        global joint positions using SMPL-style forward kinematics and showing
        each joint as a moving sphere in the Blender viewport.
        """
        import mathutils
        from mathutils import Matrix, Vector
        import bpy
        import numpy as np

        print("Generating motion preview from rots and trans...")

        kintree_table = self.kintree_table
        parents = kintree_table[0]
        num_joints = 22

        # Rest pose joint positions (in SMPL space)
        J_rest = self.J_regressor @ self.v_template  # shape (J, 3)

        # Coordinate transform: SMPL (Z-up, right-handed) → Blender (Z-up, left-handed)
        R_z = np.array([
            [0, 1, 0], 
            [-1, 0, 0], 
            [0, 0, 1]]
        )
        R_x = np.array([
            [1, 0, 0], 
            [0, 0, -1], 
            [0, 1, 0]]
        )
        R_y = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0]
        ])
        R_smpl2blender = R_y @ R_z @ R_x

        # Apply to rest joints
        J_rest_blender = (J_rest @ R_smpl2blender.T)

        # Define translation to bring min_y to 0
        # ground_offset = np.array([0, 0, -min_y])

        # Create spheres for each joint
        joint_objs = []
        for i in range(num_joints):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=J_rest_blender[i])
            sphere = bpy.context.active_object
            sphere.name = f"JointPreview_{i}"
            joint_objs.append(sphere)

        num_frames = self.num_frames

        for t in range(num_frames):
            actual_frame = t * 2
            bpy.context.scene.frame_set(actual_frame)

            # Global transforms for each joint
            global_transforms = [None] * num_joints

            # Root joint (joint 0)
            R_root_np = self.rots[t][0]  # (3, 3)
            T_root_np = self.trans[t]    # (3,)

            # Convert to Blender space
            R_root_np = R_smpl2blender @ R_root_np @ R_smpl2blender.T
            T_root_np = R_smpl2blender @ T_root_np

            R_root = Matrix(R_root_np)
            T_root = Vector(T_root_np)

            G_root = Matrix.Translation(T_root) @ R_root.to_4x4()
            global_transforms[0] = G_root

            # Other joints
            for j in range(1, num_joints):
                parent = int(parents[j])
                R_j_np = self.rots[t][j]
                offset_np = J_rest[j] - J_rest[parent]

                # Transform to Blender space
                R_j_np = R_smpl2blender @ R_j_np @ R_smpl2blender.T
                offset_np = R_smpl2blender @ offset_np

                R_j = Matrix(R_j_np)
                offset = Vector(offset_np)

                G_parent = global_transforms[parent]
                G_j = G_parent @ Matrix.Translation(offset) @ R_j.to_4x4()
                global_transforms[j] = G_j

            # Apply joint positions to spheres
            for j in range(num_joints):
                joint_pos = global_transforms[j].to_translation()
                joint_objs[j].location = joint_pos
                joint_objs[j].keyframe_insert(data_path="location", frame=actual_frame)

        print(f"Visualized SMPL motion for {num_joints} joints across {num_frames} frames.")


    def load_in_blender(self, index, mat):
        """Optional: load baked mesh only if needed."""
        return super().load_in_blender(index, mat)

    def __len__(self):
        return self.num_frames
    
    def _prepare_smplh_motion(self, rots, trans, always_on_floor=False):
        # R_x = np.array([
        #     [1, 0,  0],
        #     [0, 0, -1],
        #     [0, 1,  0]
        # ])
        # R_z = np.array([
        #     [0, 1, 0],
        #     [-1,  0, 0],
        #     [0,  0, 1]
        # ])
        # R = R_z

        # trans = trans @ R.T

        # Flip X and Y for translation
        # trans[..., 0] = -trans[..., 0]
        # trans[..., 1] = -trans[..., 1]

        # Align floor
        # if always_on_floor:
        #     trans[..., 2] -= trans[..., 2].min(1, keepdims=True)
        # else:
        # trans[..., 2] -= trans[..., 2].min()

        # Flip X and Y for rotations
        # axis_flip = np.diag([-1, -1, 1])
        # rots = axis_flip @ rots @ axis_flip.T  # matrix mult on both sides

        return rots, trans

    def load_base_armature(self):
        # prepare
        kintree_table = self.kintree_table
        parents = kintree_table[0]
        children = self.kintree_table[1] # child array

        child_map = defaultdict(list)
        for parent, child in zip(parents, children):
            if parent != -1 or parent != 4294967295:
                child_map[parent].append(child)
        
        num_joints = 22
        num_frames = self.num_frames

        # Rest pose joint positions (in SMPL space)
        J_rest = self.J_regressor @ self.v_template  # shape (J, 3)

        # Coordinate transform: SMPL (Z-up, right-handed) → Blender (Z-up, left-handed)
        R_z = np.array([
            [0, 1, 0], 
            [-1, 0, 0], 
            [0, 0, 1]]
        )
        R_x = np.array([
            [1, 0, 0], 
            [0, 0, -1], 
            [0, 1, 0]]
        )
        R_y = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0]
        ])
        R_smpl2blender = R_y @ R_z @ R_x

        # compute the global joint transformation at frame 0
        t = 0
        global_transforms = [None] * num_joints

        # Root joint (joint 0)
        R_root_np = self.rots[t][0]  # (3, 3)
        T_root_np = self.trans[t]    # (3,)

        # Convert SMPL Coodination into Blender Coordination
        R_root_np = R_smpl2blender @ R_root_np @ R_smpl2blender.T
        T_root_np = R_smpl2blender @ T_root_np

        # rots and trans at root joint
        R_root = Matrix(R_root_np)
        T_root = Vector(T_root_np)
         
        # package rots and trans then store it
        G_root = Matrix.Translation(T_root) @ R_root.to_4x4()
        global_transforms[0] = G_root

        for j in range(1, num_joints):
            parent = int(parents[j])
            R_j_np = self.rots[t][j]
            offset_np = J_rest[j] - J_rest[parent]
            R_j_np = R_smpl2blender @ R_j_np @ R_smpl2blender.T
            offset_np = R_smpl2blender @ offset_np
            R_j = Matrix(R_j_np)
            offset = Vector(offset_np)
            G_parent = global_transforms[parent]
            G_j = G_parent @ Matrix.Translation(offset) @ R_j.to_4x4()
            global_transforms[j] = G_j

        # Build armature
        armature_data = bpy.data.armatures.new("SMPLArmature")
        armature_obj = bpy.data.objects.new("SMPLArmature", armature_data)
        bpy.context.collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        # Construct Bones
        edit_bones = armature_data.edit_bones

        # Construct rest of the bones
        for j in range(1, num_joints):
            parent = int(parents[j])
            child_pos = global_transforms[j].to_translation()
            parent_pos = global_transforms[parent].to_translation()

            bone = edit_bones.new(f"Bone_{j}")
            bone.head = parent_pos
            bone.tail = child_pos
            bone.parent = edit_bones.get(f"Bone_{parent}")
            bone.use_connect = True

            self.bone_lookup[(parent, j)] = f"Bone_{j}"

        self.armature_object = armature_obj
        bpy.ops.object.mode_set(mode='OBJECT')

        # Animate
        for t in range(1, num_frames):
            actual_frame = t * 2  # optional multiplier
            bpy.context.scene.frame_set(actual_frame)

            # Recompute global transforms at frame 1
            global_transforms = [None] * num_joints

            R_root_np = self.rots[t][0]
            T_root_np = self.trans[t]
            R_root_np = R_smpl2blender @ R_root_np @ R_smpl2blender.T
            T_root_np = R_smpl2blender @ T_root_np
            R_root = Matrix(R_root_np)
            T_root = Vector(T_root_np)
            G_root = Matrix.Translation(T_root) @ R_root.to_4x4()
            global_transforms[0] = G_root

            for j in range(1, num_joints):
                parent = int(parents[j])
                R_j_np = self.rots[t][j]
                offset_np = J_rest[j] - J_rest[parent]
                R_j_np = R_smpl2blender @ R_j_np @ R_smpl2blender.T
                offset_np = R_smpl2blender @ offset_np
                R_j = Matrix(R_j_np)
                offset = Vector(offset_np)
                G_parent = global_transforms[parent]
                G_j = G_parent @ Matrix.Translation(offset) @ R_j.to_4x4()
                global_transforms[j] = G_j

            pose_bones = armature_obj.pose.bones

            for (parent, child), bone_name in self.bone_lookup.items():
                if (-1, 0) == (parent, child):
                    continue
                pose_bone = pose_bones[bone_name]
                pose_bone.rotation_mode = 'QUATERNION'

                # Extract local transform from parent
                G_parent = global_transforms[parent]
                G_child = global_transforms[child]
                local_matrix = G_parent.inverted() @ G_child

                # Extract clean location and rotation
                pose_bone.location = local_matrix.to_translation()
                pose_bone.rotation_quaternion = local_matrix.to_quaternion()

                pose_bone.keyframe_insert(data_path="location", frame=actual_frame)
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=actual_frame)