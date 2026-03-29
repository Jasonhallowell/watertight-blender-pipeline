import sys
import numpy as np  # Load Blender's numpy FIRST
sys.path.append(r"C:\Users\jason\AppData\Roaming\Python\Python311\site-packages")

import bpy
import bmesh
import os
import math
import random
from PIL import Image
import mathutils
import io
import time
from mathutils import Vector

# Use numpy as GPU fallback (cupy has numpy version conflicts with Blender)
cp = np
cp.asnumpy = lambda x: np.asarray(x)  # shim: asnumpy is identity for numpy arrays
print("Using NumPy (CPU) for computation — cupy skipped due to Blender numpy conflict")

import subprocess
import tempfile
import multiprocessing
import pickle
import shutil

# ===============================================================
# FILE PATHS (Customize as needed)
# ===============================================================
#input_fbx  = r"C:\Users\jason\Downloads\autumn-house\source\House_scene_01.fbx"
input_fbx  = r"C:\Users\jason\Documents\Chess Pieces\PelicanWithBase.fbx"
# Define the export folder and ensure it exists
export_folder = r"C:\Users\jason\Desktop\Exports"
if not os.path.exists(export_folder):
    os.makedirs(export_folder)

# Build export file paths in the export folder
output_fbx = os.path.join(export_folder, "your_baked_jetpack.obj")
TEXTURE_EXPORT_PATH = os.path.join(export_folder, "your_texture.png")
debug_before_union_fbx = os.path.join(export_folder, "debug_before_union.fbx")
debug_after_union_fbx  = os.path.join(export_folder, "debug_after_union.fbx")
debug_after_remesh_fbx = os.path.join(export_folder, "debug_after_remesh.fbx")
# New export for grid fill groups before deletion:
debug_before_grid_fill_deletion_fbx = os.path.join(export_folder, "debug_before_grid_fill_deletion.fbx")

# ===============================================================
# PART 1: Clean the original FBX (merge doubles & fill holes)
# ===============================================================
print("=== Part 1: Clean the original FBX (merge doubles & fill holes) ===")
total_part1_start = time.time()

# --- STEP 0: Clear Scene and Import FBX ---
print("Step 0: Clearing scene and importing FBX")
step0_start = time.time()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=input_fbx)
step0_end = time.time()
print(f"Step 0 completed in {step0_end - step0_start:.2f} seconds")

# Gather the imported mesh objects
mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

if not mesh_objects:
    print("No mesh objects found in the FBX file!")
else:
    # --- APPLY OBJECT TRANSFORMATIONS ---
    print("Applying object transformations (location, rotation, scale)...")
    for obj in mesh_objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.transform_apply(scale=True)
        # Uncomment the following line to apply location and rotation as well:
        # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)
    print("Object transformations applied.")

    # --- JOIN ALL MESH OBJECTS INTO ONE ---
    print("Joining all mesh parts into one object...")
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.ops.object.join()
    joined_obj = bpy.context.active_object
    mesh_objects = [joined_obj]
    print("Joined object:", joined_obj.name)

    # --- STEP 1: Process Each Mesh (Merge Vertices and Fill Holes) ---
    print("Step 1: Processing each mesh (merging vertices and filling holes)")
    step1_start = time.time()
    for obj in mesh_objects:
        print(f"  Processing mesh: {obj.name}")
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        # Merge duplicate vertices
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.00001)
        # Planar Dissolve: dissolve nearly coplanar edges/faces to simplify the mesh.
        #bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(2.0))

        # Detect boundary edges (edges with only one linked face)
        boundary_edges = {e for e in bm.edges if len(e.link_faces) == 1}
        visited_edges = set()
        edge_loops = []

        def find_loop(start_edge):
            loop_edges = [start_edge]
            visited_edges.add(start_edge)
            current_vert = start_edge.verts[1]
            while True:
                next_edge = None
                for e in boundary_edges:
                    if e not in visited_edges:
                        if current_vert in e.verts:
                            next_edge = e
                            break
                if not next_edge:
                    break
                loop_edges.append(next_edge)
                visited_edges.add(next_edge)
                current_vert = (next_edge.verts[1] if next_edge.verts[0] == current_vert else next_edge.verts[0])
                # Check if loop is closed
                if current_vert in loop_edges[0].verts:
                    return loop_edges
            return None

        for edge in boundary_edges:
            if edge not in visited_edges:
                loop = find_loop(edge)
                if loop:
                    edge_loops.append(loop)

        # --- Fill each detected loop and tag new faces ---
        # Create (or get) a custom integer layer on BMFace for grid fill groups.
        grid_fill_layer = bm.faces.layers.int.get("grid_fill_group")
        if grid_fill_layer is None:
            grid_fill_layer = bm.faces.layers.int.new("grid_fill_group")
        grid_fill_group_id = 1  # Start with group ID 1

        for loop in edge_loops:
            # Check if all edges in the loop belong to the same face.
            faces_in_loop = {f for edge in loop for f in edge.link_faces}
            if len(faces_in_loop) == 1:
                print("Skipping grid fill for loop that belongs to a single face.")
                continue

            bpy.ops.mesh.select_all(action='DESELECT')
            for edge in loop:
                edge.select = True

            # Record current faces before grid fill
            old_faces = set(bm.faces)
            #bpy.ops.mesh.fill_grid(span=1, offset=0, use_interp_simple=False)
            # Only tag new faces (the ones not present in old_faces and still selected)
            new_faces = [f for f in bm.faces if f not in old_faces and f.select]
            for f in new_faces:
                f[grid_fill_layer] = grid_fill_group_id

            grid_fill_group_id += 1

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
    step1_end = time.time()
    print(f"Step 1 completed in {step1_end - step1_start:.2f} seconds")

    # --- STEP 2: Separate Loose Parts into Individual Objects ---
    print("Step 2: Separating loose parts into individual objects")
    step2_start = time.time()
    for obj in mesh_objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)
    separated_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    print(f"  Separated into {len(separated_objects)} objects.")
    step2_end = time.time()
    print(f"Step 2 completed in {step2_end - step2_start:.2f} seconds")

    # --- STEP 2.5: Mark overlapping coplanar faces in pink (debug) ---
    print("Step 2.5: Marking debug faces in pink (overlapping coplanar) and purple (intersecting)")

    def mark_overlapping_faces(obj, distance_threshold=0.001, normal_dot_threshold=0.99, epsilon=0.0001):
        # Create (or get) a debug pink material
        pink_mat = None
        for mat in bpy.data.materials:
            if mat.name == "Debug_Pink":
                pink_mat = mat
                break
        if pink_mat is None:
            pink_mat = bpy.data.materials.new("Debug_Pink")
            pink_mat.diffuse_color = (1.0, 0.0, 1.0, 1.0)
        if pink_mat.name not in [m.name for m in obj.data.materials]:
            obj.data.materials.append(pink_mat)
        pink_index = obj.data.materials.find(pink_mat.name)

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bvh = mathutils.bvhtree.BVHTree.FromBMesh(bm)

        for face in bm.faces:
            center = face.calc_center_median()
            normal = face.normal
            # Start the ray behind the face
            ray_origin = center - normal * epsilon
            remaining_dist = distance_threshold

            # Try several times if the ray hits the same face
            for _ in range(5):
                hit = bvh.ray_cast(ray_origin, normal, remaining_dist)
                if hit[0] is None:
                    break
                hit_loc, hit_normal, hit_face_index, hit_distance = hit
                if hit_face_index == face.index:
                    # The ray hit the originating face; shift the origin forward and try again.
                    ray_origin = hit_loc + normal * epsilon
                    remaining_dist -= hit_distance + epsilon
                    if remaining_dist <= 0:
                        break
                    continue
                if normal.dot(hit_normal) >= normal_dot_threshold:
                    face.material_index = pink_index
                    try:
                        bm.faces[hit_face_index].material_index = pink_index
                    except Exception as e:
                        print(f"Warning: Could not mark face {hit_face_index}: {e}")
                break

        bm.to_mesh(obj.data)
        bm.free()

    def mark_intersecting_faces(obj, epsilon=1e-4):
        # Create (or get) a debug purple material
        purple_mat = None
        for mat in bpy.data.materials:
            if mat.name == "Debug_Purple":
                purple_mat = mat
                break
        if purple_mat is None:
            purple_mat = bpy.data.materials.new("Debug_Purple")
            purple_mat.diffuse_color = (0.5, 0.0, 0.5, 1.0)
        if purple_mat.name not in [m.name for m in obj.data.materials]:
            obj.data.materials.append(purple_mat)
        purple_index = obj.data.materials.find(purple_mat.name)

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        # Build fan-triangulation for each face and record its vertex set.
        face_tris = {}
        face_verts = {}
        for face in bm.faces:
            verts = [loop.vert for loop in face.loops]
            face_verts[face] = set(verts)
            if len(verts) < 3:
                continue
            tris = []
            v0 = verts[0].co.copy()
            for i in range(1, len(verts) - 1):
                v1 = verts[i].co.copy()
                v2 = verts[i+1].co.copy()
                area = 0.5 * (v1 - v0).cross(v2 - v0).length
                if area < 1e-8:
                    continue
                tris.append((v0, v1, v2))
            if tris:
                face_tris[face] = tris

        face_list = list(face_tris.keys())
        n = len(face_list)
        intersecting_faces = set()

        # Iterate over all pairs of faces.
        for i in range(n):
            face_i = face_list[i]
            tris_i = face_tris[face_i]
            xs_i = [v.x for tri in tris_i for v in tri]
            ys_i = [v.y for tri in tris_i for v in tri]
            zs_i = [v.z for tri in tris_i for v in tri]
            bbox_i = (min(xs_i), max(xs_i), min(ys_i), max(ys_i), min(zs_i), max(zs_i))
            for j in range(i + 1, n):
                face_j = face_list[j]
                # Skip if faces share vertices (likely adjacent)
                if face_verts[face_i] & face_verts[face_j]:
                    continue
                tris_j = face_tris[face_j]
                xs_j = [v.x for tri in tris_j for v in tri]
                ys_j = [v.y for tri in tris_j for v in tri]
                zs_j = [v.z for tri in tris_j for v in tri]
                bbox_j = (min(xs_j), max(xs_j), min(ys_j), max(ys_j), min(zs_j), max(zs_j))
                # Quick bounding-box check.
                def bbox_overlap(b1, b2):
                    return not (b1[1] < b2[0] or b2[1] < b1[0] or
                                b1[3] < b2[2] or b2[3] < b1[2] or
                                b1[5] < b2[4] or b2[5] < b1[4])
                if not bbox_overlap(bbox_i, bbox_j):
                    continue
                found = False
                for tri_i in tris_i:
                    for tri_j in tris_j:
                        # Use the new triangles_intersect with an overlap tolerance.
                        if triangles_intersect(tri_i, tri_j, epsilon=1e-4, overlap_tol=1e-3):
                            intersecting_faces.add(face_i)
                            intersecting_faces.add(face_j)
                            found = True
                            break
                    if found:
                        break

        # Mark intersecting faces with the purple material.
        for face in intersecting_faces:
            face.material_index = purple_index

        bm.to_mesh(obj.data)
        bm.free()

    # New unified triangles_intersect using an overlap tolerance.
    def triangles_intersect(tri1, tri2, epsilon=1e-4, overlap_tol=1e-3):
        # Helper: project a triangle onto an axis.
        def project_triangle(tri, axis):
            dots = [v.dot(axis) for v in tri]
            return min(dots), max(dots)
        # Helper: test if projections overlap on the given axis with tolerance.
        def axis_test(axis, tri1, tri2):
            if axis.length < epsilon:
                return True  # Skip degenerate axis.
            axis = axis.normalized()
            min1, max1 = project_triangle(tri1, axis)
            min2, max2 = project_triangle(tri2, axis)
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < overlap_tol:
                return False
            return True

        axes = []
        # Normals of each triangle.
        edge1 = tri1[1] - tri1[0]
        edge2 = tri1[2] - tri1[0]
        axes.append(edge1.cross(edge2))
        edge1 = tri2[1] - tri2[0]
        edge2 = tri2[2] - tri2[0]
        axes.append(edge1.cross(edge2))
        # Cross products of edges.
        edges1 = [tri1[1] - tri1[0], tri1[2] - tri1[1], tri1[0] - tri1[2]]
        edges2 = [tri2[1] - tri2[0], tri2[2] - tri2[1], tri2[0] - tri2[2]]
        for e1 in edges1:
            for e2 in edges2:
                axes.append(e1.cross(e2))
        for axis in axes:
            if not axis_test(axis, tri1, tri2):
                return False
        return True

    # Now run both debug marking routines on each separated mesh object:
    for obj in separated_objects:
        if obj.type == 'MESH':
            mark_overlapping_faces(obj, distance_threshold=0.0001, normal_dot_threshold=0.99)
            mark_intersecting_faces(obj, epsilon=1e-4)

    print("Step 2.5 completed.")

    # --- STEP 2.6: Delete grid fill groups with pink or purple faces ---
    print("Step 2.6: Deleting grid fill groups with pink or purple faces")
    # Export debug FBX before deletion of grid fill groups
    bpy.ops.object.select_all(action='DESELECT')
    for obj in separated_objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.export_scene.fbx(
        filepath=debug_before_grid_fill_deletion_fbx,
        use_selection=True,
        axis_forward='-Z',
        axis_up='Y'
    )
    print(f"Exported debug FBX before deletion of grid fill groups to: {debug_before_grid_fill_deletion_fbx}")

    for obj in separated_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)
            grid_fill_layer = bm.faces.layers.int.get("grid_fill_group")
            if grid_fill_layer:
                # Group faces by grid fill group id (skip default value 0)
                groups = {}
                for face in bm.faces:
                    group_val = face[grid_fill_layer]
                    if group_val != 0:
                        groups.setdefault(group_val, []).append(face)
                # Determine the Debug_Pink and Debug_Purple material indices
                pink_index = -1
                purple_index = -1
                for mat in obj.data.materials:
                    if mat.name == "Debug_Pink":
                        pink_index = obj.data.materials.find(mat.name)
                    elif mat.name == "Debug_Purple":
                        purple_index = obj.data.materials.find(mat.name)
                # Delete groups if any face is marked with Debug_Pink or Debug_Purple
                for group_id, faces in groups.items():
                    if any(face.material_index == pink_index or face.material_index == purple_index for face in faces):
                        print(f"  Deleting grid fill group {group_id} with debug faces on {obj.name}")
                        bmesh.ops.delete(bm, geom=faces, context='FACES')
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.object.mode_set(mode='OBJECT')

    # --- STEP 2.7: Solidify objects with boundary edges ---
    print("Step 2.7: Solidifying objects with boundary edges")
    for obj in separated_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)
            # Identify boundary edges (edges with only one linked face)
            boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
            if boundary_edges:
                bpy.ops.object.mode_set(mode='OBJECT')
                print(f"  {obj.name} has {len(boundary_edges)} boundary edges. Applying solidify...")
                mod = obj.modifiers.new(name="Solidify_Boundary", type='SOLIDIFY')
                mod.thickness = 0.001  # Adjust thickness as needed
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier=mod.name)
            else:
                bpy.ops.object.mode_set(mode='OBJECT')
            bm.free()

    # --- STEP 3: Single Collection-Based Boolean Union ---
    print("Step 3: Performing single collection-based Boolean Union")
    step3_start = time.time()
    final_objects = []
    if len(separated_objects) <= 1:
        final_objects = separated_objects
    else:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in separated_objects:
            obj.select_set(True)
        bpy.ops.export_scene.fbx(
            filepath=debug_before_union_fbx,
            use_selection=True,
            axis_forward='-Z',
            axis_up='Y'
        )
        print(f"Exported debug FBX before union to: {debug_before_union_fbx}")
        bpy.ops.object.select_all(action='DESELECT')
        print("Recalculating normals for all separated objects...")
        for obj in separated_objects:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            obj.select_set(False)
        print("Normals recalculated.")

        bool_collection = bpy.data.collections.new("BooleanUnionCollection")
        bpy.context.scene.collection.children.link(bool_collection)
        base_obj = separated_objects[0]
        bpy.context.view_layer.objects.active = base_obj
        for obj in separated_objects:
            bool_collection.objects.link(obj)
            if obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(obj)
        mod = base_obj.modifiers.new(name="UnionAll", type='BOOLEAN')
        mod.operation = 'UNION'
        mod.solver = 'EXACT'
        mod.operand_type = 'COLLECTION'
        mod.collection = bool_collection
        mod.use_self = True
        mod.material_mode = 'TRANSFER'
        bpy.ops.object.modifier_apply(modifier=mod.name)
        for obj in separated_objects[1:]:
            if obj != base_obj:
                bpy.data.objects.remove(obj, do_unlink=True)
        final_objects = [base_obj]
        bpy.ops.object.select_all(action='DESELECT')
        base_obj.select_set(True)
        bpy.ops.export_scene.fbx(
            filepath=debug_after_union_fbx,
            use_selection=True,
            axis_forward='-Z',
            axis_up='Y'
        )
        print(f"Exported debug FBX after union to: {debug_after_union_fbx}")
    step3_end = time.time()
    print(f"Step 3 completed in {step3_end - step3_start:.2f} seconds")

    bpy.ops.object.select_all(action='DESELECT')
    for obj in final_objects:
        obj.select_set(True)

    total_part1_end = time.time()
    print(f"Part 1 completed successfully in {total_part1_end - total_part1_start:.2f} seconds.")


# ===============================================================
# PART 2: Texture + Raycast Color Transfer (limited palette)
# ===============================================================
OBJ_EXPORT_PATH = output_fbx

# Clustering parameters:
K                  = 15   # Number of color clusters
MAX_ITERATIONS     = 10
MAX_SAMPLING_SIZE  = 200000

REF_X = 0.95047
REF_Y = 1.00000
REF_Z = 1.08883

# ---------------------------
# HELPER FUNCTIONS (CPU)
# ---------------------------
def clamp_int(v, low, high):
    return max(low, min(v, high))

def srgb_to_linear(c):
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4

def linear_to_srgb(f):
    if f <= 0.0031308:
        return 12.92 * f
    else:
        return 1.055 * (f ** (1.0 / 2.4)) - 0.055

def get_image_as_pil(image):
    im_path = bpy.path.abspath(image.filepath)
    if os.path.isfile(im_path):
        print(f"Using image from disk: {im_path}")
        return Image.open(im_path).convert("RGBA")
    elif image.packed_file is not None:
        print(f"Using packed image data for: {image.name}")
        packed_data = image.packed_file.data
        im = Image.open(io.BytesIO(packed_data)).convert("RGBA")
        try:
            im.save(TEXTURE_EXPORT_PATH)
            print(f"Exported packed texture to: {TEXTURE_EXPORT_PATH}")
        except Exception as e:
            print(f"Failed to export packed texture: {e}")
        return im
    else:
        raise RuntimeError(f"Image file not found and no packed data: {im_path}")

def barycentric_coords_2d(px, py, uvA, uvB, uvC):
    xA, yA = uvA
    xB, yB = uvB
    xC, yC = uvC
    denom = (yB - yC)*(xA - xC) + (xC - xB)*(yA - yC)
    if abs(denom) < 1e-14:
        return None
    alpha = ((yB - yC)*(px - xC) + (xC - xB)*(py - yC)) / denom
    beta  = ((yC - yA)*(px - xC) + (xA - xC)*(py - yC)) / denom
    gamma = 1.0 - alpha - beta
    return (alpha, beta, gamma)

def is_inside_triangle(a, b, c, eps=1e-8):
    return -eps <= a <= 1.0+eps and -eps <= b <= 1.0+eps and -eps <= c <= 1.0+eps

# ---------------------------
# OPTIMIZED PIXEL GATHERING (Modified for Multiple Textures)
# ---------------------------
def gather_used_pixels_optimized(obj, material_to_pil, max_sample=200000):
    print("  Starting optimized pixel gathering for multiple textures...")
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    if not uv_layer:
        bm.free()
        return []

    used_pixels = []
    total_uv_area = 0
    # First pass: compute total UV area for faces with valid texture
    for face in bm.faces:
        if len(face.loops) < 3:
            continue
        if face.material_index not in material_to_pil:
            continue
        uvs = [l[uv_layer].uv for l in face.loops]
        face_area = 0
        for i in range(1, len(uvs) - 1):
            a, b, c = uvs[0], uvs[i], uvs[i+1]
            area = 0.5 * abs((b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y))
            face_area += area
        total_uv_area += face_area

    target_sample_density = max_sample / total_uv_area if total_uv_area > 0 else 0
    print(f"  Target sample density: {target_sample_density:.6f}")
    total_faces = len(bm.faces)
    processed_faces = 0
    start_time = time.time()
    last_report_time = start_time
    pixel_set = set()

    for face in bm.faces:
        if len(face.loops) < 3:
            continue
        if face.material_index not in material_to_pil:
            continue

        pil_image = material_to_pil[face.material_index]
        w, h = pil_image.size

        loops = list(face.loops)
        uv_points = [l[uv_layer].uv for l in face.loops]
        uv_pixels = [(uv.x * w, (1.0 - uv.y) * h) for uv in uv_points]
        min_x = max(0, int(math.floor(min(p[0] for p in uv_pixels))))
        max_x = min(w - 1, int(math.ceil(max(p[0] for p in uv_pixels))))
        min_y = max(0, int(math.floor(min(p[1] for p in uv_pixels))))
        max_y = min(h - 1, int(math.ceil(max(p[1] for p in uv_pixels))))

        face_area = 0
        for i in range(1, len(uv_points) - 1):
            a, b, c = uv_points[0], uv_points[i], uv_points[i+1]
            area = 0.5 * abs((b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y))
            face_area += area

        area_ratio = face_area / total_uv_area if total_uv_area > 0 else 0
        target_samples_in_face = max(10, int(area_ratio * max_sample))
        face_pixels = (max_x - min_x + 1) * (max_y - min_y + 1)
        sample_step = max(1, int(math.sqrt(face_pixels / target_samples_in_face))) if face_pixels > 0 else 1

        for py in range(min_y, max_y + 1, sample_step):
            for px in range(min_x, max_x + 1, sample_step):
                pixel_key = (px, py)
                if pixel_key in pixel_set:
                    continue
                is_inside = False
                for i in range(1, len(loops) - 1):
                    uv_tri = [(loops[0][uv_layer].uv.x * w, (1.0 - loops[0][uv_layer].uv.y) * h),
                              (loops[i][uv_layer].uv.x * w, (1.0 - loops[i][uv_layer].uv.y) * h),
                              (loops[i+1][uv_layer].uv.x * w, (1.0 - loops[i+1][uv_layer].uv.y) * h)]
                    p0x, p0y = uv_tri[0]
                    p1x, p1y = uv_tri[1]
                    p2x, p2y = uv_tri[2]
                    e0 = (px - p0x) * (p1y - p0y) - (py - p0y) * (p1x - p0x)
                    e1 = (px - p1x) * (p2y - p1y) - (py - p1y) * (p2x - p1x)
                    e2 = (px - p2x) * (p0y - p2y) - (py - p2y) * (p0x - p2x)
                    if (e0 >= 0 and e1 >= 0 and e2 >= 0) or (e0 <= 0 and e1 <= 0 and e2 <= 0):
                        is_inside = True
                        break

                if is_inside:
                    pixel_set.add(pixel_key)
                    R, G, B, A = pil_image.getpixel((px, py))
                    R_lin = srgb_to_linear(R / 255.0)
                    G_lin = srgb_to_linear(G / 255.0)
                    B_lin = srgb_to_linear(B / 255.0)
                    used_pixels.append((R_lin, G_lin, B_lin))
                    if len(used_pixels) >= max_sample:
                        print(f"  Early termination: gathered {len(used_pixels)} samples")
                        bm.free()
                        return used_pixels

        processed_faces += 1
        current_time = time.time()
        if current_time - last_report_time > 2:
            progress = (processed_faces / total_faces) * 100
            elapsed = current_time - start_time
            estimated_total = elapsed / (processed_faces / total_faces) if processed_faces > 0 else 0
            remaining = max(0, estimated_total - elapsed)
            pixel_count = len(used_pixels)
            print(f"  Progress: {processed_faces}/{total_faces} faces ({progress:.1f}%)")
            print(f"  Samples gathered: {pixel_count}/{max_sample}")
            print(f"  Estimated time remaining: {remaining:.1f} seconds")
            last_report_time = current_time

    bm.free()
    if len(used_pixels) > max_sample:
        used_pixels = random.sample(used_pixels, max_sample)
    print(f"  Gathered {len(used_pixels)} pixel samples")
    return used_pixels

# ---------------------------
# HELPER FUNCTIONS (GPU - CuPy)
# ---------------------------
def cupy_srgb_to_linear(arr_cp):
    return cp.where(arr_cp <= 0.04045, arr_cp / 12.92, ((arr_cp + 0.055) / 1.055) ** 2.4)

def linear_rgb_to_lab_cp(rgb_lin_cp):
    r = rgb_lin_cp[..., 0]
    g = rgb_lin_cp[..., 1]
    b = rgb_lin_cp[..., 2]
    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    X /= REF_X
    Y /= REF_Y
    Z /= REF_Z
    def pivot(x):
        return cp.where(x > 0.008856, cp.cbrt(x), 7.787*x + 16/116)
    Xp = pivot(X)
    Yp = pivot(Y)
    Zp = pivot(Z)
    L = 116*Yp - 16
    A = 500*(Xp - Yp)
    Bv = 200*(Yp - Zp)
    return cp.stack([L, A, Bv], axis=-1)

def lab_to_linear_rgb_cp(lab_cp):
    L = lab_cp[..., 0]
    A = lab_cp[..., 1]
    Bv = lab_cp[..., 2]
    Y_n = (L + 16)/116
    X_n = A/500 + Y_n
    Z_n = Y_n - Bv/200
    def inv_pivot(x):
        return cp.where(
            x**3 > 0.008856,
            x**3,
            (x - 16/116)/7.787
        )
    X = REF_X * inv_pivot(X_n)
    Y = REF_Y * inv_pivot(Y_n)
    Z = REF_Z * inv_pivot(Z_n)
    r_lin =  3.2406*X - 1.5372*Y - 0.4986*Z
    g_lin = -0.9689*X + 1.8758*Y + 0.0415*Z
    b_lin =  0.0557*X - 0.2040*Y + 1.0570*Z
    rgb_lin = cp.stack([r_lin, g_lin, b_lin], axis=-1)
    rgb_lin = cp.clip(rgb_lin, 0, 1)
    return rgb_lin

# ---------------------------
# OPTIMIZED K-MEANS CLUSTERING
# ---------------------------
def run_kmeans_on_pixels_lab_optimized(pixel_list_lin, K=8, max_iter=10):
    print("  Starting optimized K-means clustering...")
    if not pixel_list_lin:
        return np.zeros((0, 3), dtype=np.float64)
    pixel_array = np.array(pixel_list_lin, dtype=np.float64)
    n_samples = pixel_array.shape[0]
    if n_samples == 0:
        return np.zeros((0, 3), dtype=np.float64)
    MAX_GPU_SAMPLES = 500000
    use_batching = n_samples > MAX_GPU_SAMPLES
    batch_size = min(MAX_GPU_SAMPLES, n_samples)
    print(f"  Total samples: {n_samples}")
    print(f"  Using {'batched' if use_batching else 'full'} processing")
    print(f"  K={K}, max_iterations={max_iter}")
    start_time = time.time()
    if use_batching:
        subsample_indices = np.random.choice(n_samples, batch_size, replace=False)
        subsample = pixel_array[subsample_indices]
        data_cp = cp.array(subsample)
        data_lab_cp = linear_rgb_to_lab_cp(data_cp)
        centroids_cp = cp.empty((K, 3), dtype=cp.float64)
        idx = cp.random.randint(batch_size)
        centroids_cp[0] = data_lab_cp[idx]
        for k in range(1, K):
            print(f"  Initializing centroid {k+1}/{K}")
            dists = cp.min(cp.sum((data_lab_cp[:, None, :] - centroids_cp[None, :k, :])**2, axis=2), axis=1)
            probs = dists / cp.sum(dists)
            idx = cp.random.choice(batch_size, size=1, p=cp.asnumpy(probs))[0]
            centroids_cp[k] = data_lab_cp[idx]
        for iteration in range(max_iter):
            print(f"  K-means iteration {iteration+1}/{max_iter}")
            new_centroids_cp = cp.zeros_like(centroids_cp)
            counts_cp = cp.zeros(K, dtype=cp.int32)
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch = pixel_array[batch_start:batch_end]
                batch_cp = cp.array(batch)
                batch_lab_cp = linear_rgb_to_lab_cp(batch_cp)
                distances = cp.sum((batch_lab_cp[:, None, :] - centroids_cp[None, :, :])**2, axis=2)
                labels = cp.argmin(distances, axis=1)
                for k in range(K):
                    mask = (labels == k)
                    if cp.any(mask):
                        new_centroids_cp[k] += cp.sum(batch_lab_cp[mask], axis=0)
                        counts_cp[k] += cp.sum(mask)
            for k in range(K):
                if counts_cp[k] > 0:
                    new_centroids_cp[k] /= counts_cp[k]
                else:
                    idx = cp.random.randint(batch_size)
                    new_centroids_cp[k] = data_lab_cp[idx]
            delta = cp.sum((new_centroids_cp - centroids_cp)**2)
            centroids_cp = new_centroids_cp
            print(f"    Iteration delta={float(delta):.6f}")
            if delta < 1e-6:
                print(f"  Converged after {iteration+1} iterations!")
                break
    else:
        data_cp = cp.array(pixel_array)
        data_lab_cp = linear_rgb_to_lab_cp(data_cp)
        centroids_cp = cp.empty((K, 3), dtype=cp.float64)
        idx = cp.random.randint(n_samples)
        centroids_cp[0] = data_lab_cp[idx]
        for k in range(1, K):
            print(f"  Initializing centroid {k+1}/{K}")
            dists = cp.min(cp.sum((data_lab_cp[:, None, :] - centroids_cp[None, :k, :])**2, axis=2), axis=1)
            probs = dists / cp.sum(dists)
            idx = cp.random.choice(n_samples, size=1, p=cp.asnumpy(probs))[0]
            centroids_cp[k] = data_lab_cp[idx]
        for iteration in range(max_iter):
            print(f"  K-means iteration {iteration+1}/{max_iter}")
            distances = cp.sum((data_lab_cp[:, None, :] - centroids_cp[None, :, :])**2, axis=2)
            labels = cp.argmin(distances, axis=1)
            new_centroids_cp = cp.zeros_like(centroids_cp)
            for k in range(K):
                mask = (labels == k)
                if cp.any(mask):
                    new_centroids_cp[k] = cp.mean(data_lab_cp[mask], axis=0)
                else:
                    new_centroids_cp[k] = centroids_cp[k]
            delta = cp.sum((new_centroids_cp - centroids_cp)**2)
            centroids_cp = new_centroids_cp
            print(f"    Iteration delta={float(delta):.6f}")
            if delta < 1e-6:
                print(f"  Converged after {iteration+1} iterations!")
                break
    centroids_lin_cp = lab_to_linear_rgb_cp(centroids_cp)
    centroids_lin = cp.asnumpy(centroids_lin_cp)
    end_time = time.time()
    print(f"  K-means clustering completed in {end_time - start_time:.2f} seconds")
    return centroids_lin

# ---------------------------
# CPU Helper: convert between linear RGB <-> LAB (NumPy)
# ---------------------------
def linear_rgb_to_lab_np(rgb_lin):
    r = rgb_lin[..., 0]
    g = rgb_lin[..., 1]
    b = rgb_lin[..., 2]
    X = r*0.4124 + g*0.3576 + b*0.1805
    Y = r*0.2126 + g*0.7152 + b*0.0722
    Z = r*0.0193 + g*0.1192 + b*0.9505
    X /= REF_X
    Y /= REF_Y
    Z /= REF_Z
    def pivot(v):
        return np.where(v > 0.008856, np.cbrt(v), 7.787*v + 16/116)
    Xp = pivot(X)
    Yp = pivot(Y)
    Zp = pivot(Z)
    L = 116*Yp - 16
    A = 500*(Xp - Yp)
    Bv = 200*(Yp - Zp)
    return np.stack([L, A, Bv], axis=-1)

def lab_to_linear_rgb_np(lab):
    L = lab[..., 0]
    A = lab[..., 1]
    Bv = lab[..., 2]
    Y_n = (L + 16)/116
    X_n = A/500 + Y_n
    Z_n = Y_n - Bv/200
    def inv_pivot(x):
        return np.where(
            x**3 > 0.008856,
            x**3,
            (x - 16/116)/7.787
        )
    X = REF_X * inv_pivot(X_n)
    Y = REF_Y * inv_pivot(Y_n)
    Z = REF_Z * inv_pivot(Z_n)
    r_lin =  3.2406*X - 1.5372*Y - 0.4986*Z
    g_lin = -0.9689*X + 1.8758*Y + 0.0415*Z
    b_lin =  0.0557*X - 0.2040*Y + 1.0570*Z
    rgb_lin = np.stack([r_lin, g_lin, b_lin], axis=-1)
    return np.clip(rgb_lin, 0, 1)

def convert_centroids_lin_to_lab(centroids_lin):
    return linear_rgb_to_lab_np(centroids_lin)

def find_nearest_centroid_index(color_lin, centroids_lin, centroids_lab):
    color_lin_arr = np.array(color_lin, dtype=np.float64).reshape((1, 3))
    color_lab = linear_rgb_to_lab_np(color_lin_arr)[0]
    diff = centroids_lab - color_lab
    dist_sq = np.sum(diff**2, axis=1)
    return np.argmin(dist_sq)

def create_materials_from_centroids(centroids_lin):
    materials = []
    for i_c, c_lin in enumerate(centroids_lin):
        mat = bpy.data.materials.new(f"Cluster_{i_c}")
        mat.use_nodes = True
        nt = mat.node_tree
        for nd in nt.nodes:
            nt.nodes.remove(nd)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        outp = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs['BSDF'], outp.inputs['Surface'])
        R_s = linear_to_srgb(c_lin[0])
        G_s = linear_to_srgb(c_lin[1])
        B_s = linear_to_srgb(c_lin[2])
        bsdf.inputs["Base Color"].default_value = (R_s, G_s, B_s, 1.0)
        materials.append((mat, c_lin))
    return materials

# ===============================================================
# RAY-CAST + SKIP NON-UV'd FACES - OPTIMIZED (Modified for Multiple Textures)
# ===============================================================
def ray_cast_for_colored_face(bvh, origin_world, direction_world,
                              bm_orig, uv_layer_orig,
                              material_to_pil,
                              max_dist=1e10, max_steps=10):
    EPSILON = 1e-4
    remain_dist = max_dist
    cur_origin = origin_world.copy()
    for _ in range(max_steps):
        hit = bvh.ray_cast(cur_origin, direction_world, remain_dist)
        if hit[0] is None:
            return None
        hit_location, hit_normal, hit_face_index, hit_distance = hit
        next_origin = hit_location + direction_world * EPSILON
        color_lin = get_uv_color_from_face(bm_orig, uv_layer_orig, hit_face_index,
                                           hit_location, material_to_pil)
        if color_lin is not None:
            return (hit_location, color_lin)
        cur_origin = next_origin
        remain_dist -= (hit_distance + EPSILON)
        if remain_dist <= 0:
            break
    return None

def get_uv_color_from_face(bm_orig, uv_layer, face_idx, hit_location_world, material_to_pil):
    if face_idx < 0 or face_idx >= len(bm_orig.faces):
        return None
    face = bm_orig.faces[face_idx]
    if len(face.loops) < 3:
        return None
    # Use the texture corresponding to the face's material
    if face.material_index not in material_to_pil:
        return None
    image_pil = material_to_pil[face.material_index]
    w, h = image_pil.size
    local_hit = hit_location_world.copy()
    loops = list(face.loops)
    A_co = loops[0].vert.co
    A_uv = loops[0][uv_layer].uv.copy()
    for i in range(1, len(loops) - 1):
        B_co = loops[i].vert.co
        C_co = loops[i+1].vert.co
        B_uv = loops[i][uv_layer].uv.copy()
        C_uv = loops[i+1][uv_layer].uv.copy()
        bc = barycentric_3d(local_hit, A_co, B_co, C_co)
        if bc is None:
            continue
        a, b, c = bc
        if a >= 0 and b >= 0 and c >= 0 and (a + b + c <= 1.00001):
            uvx = A_uv.x*a + B_uv.x*b + C_uv.x*c
            uvy = A_uv.y*a + B_uv.y*b + C_uv.y*c
            return sample_texture_at_uv(uvx, uvy, image_pil, w, h)
    return None

def barycentric_3d(p, a, b, c, eps=1e-14):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        return None
    inv_denom = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * inv_denom
    w = (d00 * d21 - d01 * d20) * inv_denom
    u = 1.0 - v - w
    return (u, v, w)

def sample_texture_at_uv(u, v, image_pil, w, h):
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return None
    px = int(u * w)
    py = int((1 - v) * h)
    if px < 0 or px >= w or py < 0 or py >= h:
        return None
    R_s, G_s, B_s, A_s = image_pil.getpixel((px, py))
    R_lin = srgb_to_linear(R_s / 255.0)
    G_lin = srgb_to_linear(G_s / 255.0)
    B_lin = srgb_to_linear(B_s / 255.0)
    return (R_lin, G_lin, B_lin)

def ray_cast_through_face(bvh, face_center, face_normal, bm_orig, uv_layer_orig, material_to_pil):
    INSIDE_OFFSET = 0.0001
    MAX_DIST = 1.0
    CLOSE_ENOUGH = 0.0001
    MAX_STEPS = 6
    start_pos = face_center - (face_normal * INSIDE_OFFSET)
    hit_info = ray_cast_for_colored_face(
        bvh, start_pos, face_normal,
        bm_orig, uv_layer_orig, material_to_pil,
        max_dist=MAX_DIST, max_steps=MAX_STEPS
    )
    if hit_info:
        hit_location, color = hit_info
        distance = (hit_location - face_center).length
        if distance < CLOSE_ENOUGH:
            return hit_info
    opposite_hit = ray_cast_for_colored_face(
        bvh, start_pos, -face_normal,
        bm_orig, uv_layer_orig, material_to_pil,
        max_dist=MAX_DIST, max_steps=MAX_STEPS
    )
    if hit_info and opposite_hit:
        hit_loc, _ = hit_info
        opp_loc, _ = opposite_hit
        dist = (hit_loc - face_center).length
        opp_dist = (opp_loc - face_center).length
        return hit_info if dist < opp_dist else opposite_hit
    return hit_info or opposite_hit

# ---------------------------
# GPU-BASED Stage 2 - F: Assign Clusters via Ray Casting (Modified for Multiple Textures)
# ---------------------------
def step_F_gpu(remesh_obj, bvh, bm_orig, uv_layer_orig, material_to_pil,
               cluster_centroids_lin, cluster_centroids_lab):
    print("Stage 2 - F: Assigning clusters via GPU-based vectorized processing")
    stepF_start = time.time()
    face_indices = []
    hit_colors = []  # For each face, store the hit color (R_lin, G_lin, B_lin) or None.
    total_faces = len(remesh_obj.data.polygons)
    last_report = time.time()
    for fi, face in enumerate(remesh_obj.data.polygons):
        face_indices.append(face.index)
        face_center_world = remesh_obj.matrix_world @ face.center
        face_normal_world = remesh_obj.matrix_world.to_3x3() @ face.normal
        face_normal_world.normalize()
        colored_hit = ray_cast_through_face(
            bvh, face_center_world, face_normal_world,
            bm_orig, uv_layer_orig, material_to_pil
        )
        if colored_hit:
            _, color = colored_hit
            hit_colors.append(color)
        else:
            hit_colors.append(None)
        now = time.time()
        if now - last_report > 5:
            print(f"  Ray casting: {fi+1}/{total_faces} faces ({(fi+1)/total_faces*100:.1f}%)")
            last_report = now
    valid_face_indices = []
    valid_colors = []
    for idx, color in zip(face_indices, hit_colors):
        if color is not None:
            valid_face_indices.append(idx)
            valid_colors.append(color)
    material_assignments = {}
    if valid_colors:
        colors_cp = cp.array(valid_colors, dtype=cp.float64)
        colors_lab_cp = linear_rgb_to_lab_cp(colors_cp)
        centroids_lab_cp = cp.array(cluster_centroids_lab, dtype=cp.float64)
        dists = cp.sum((colors_lab_cp[:, None, :] - centroids_lab_cp[None, :, :]) ** 2, axis=2)
        nearest_indices = cp.asnumpy(cp.argmin(dists, axis=1)).astype(int).tolist()
        for face_idx, mat_idx in zip(valid_face_indices, nearest_indices):
            material_assignments[face_idx] = mat_idx
    for idx in face_indices:
        if idx not in material_assignments:
            material_assignments[idx] = 0
    for face in remesh_obj.data.polygons:
        face.material_index = material_assignments.get(face.index, 0)
    stepF_end = time.time()
    print(f"Stage 2 - F completed in {stepF_end - stepF_start:.2f} seconds")

# ---------------------------
# MAIN STAGE 2
# ---------------------------
def main_stage2():
    print("=== Part 2: Texture-based Color Transfer with Limited Palette ===")
    total_stage2_start = time.time()
    print("Stage 2 - A: Acquiring final object and texture images")
    stepA_start = time.time()
    obj_list = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not obj_list:
        raise RuntimeError("No mesh objects found in the current scene.")
    source_obj = obj_list[0]

    # Build mapping from material slot index to its texture (PIL image)
    material_to_pil = {}
    for i, mat in enumerate(source_obj.data.materials):
        if mat and mat.use_nodes:
            for nd in mat.node_tree.nodes:
                if nd.type == 'TEX_IMAGE' and nd.image:
                    try:
                        pil_im = get_image_as_pil(nd.image)
                        material_to_pil[i] = pil_im
                        print(f"  Material {mat.name} (index {i}) uses texture: {nd.image.name}")
                        break  # Use the first found texture for this material
                    except Exception as e:
                        print(f"Failed to load image for material index {i}: {e}")
    if not material_to_pil:
        raise RuntimeError("No texture images found in any material nodes on the final object.")
    stepA_end = time.time()
    print(f"Stage 2 - A completed in {stepA_end - stepA_start:.2f} seconds")

    print("Stage 2 - B: (Skipping single texture loading as multiple textures are used)")

    print("Stage 2 - C: Gathering used pixels and running K-means color clustering")
    stepC_start = time.time()
    used_pixels = gather_used_pixels_optimized(source_obj, material_to_pil, max_sample=MAX_SAMPLING_SIZE)
    if not used_pixels:
        raise RuntimeError("No used pixels found in the object's UV space.")
    print(f"  Total used pixels gathered: {len(used_pixels)}")
    print(f"  Running optimized GPU-based K-means with K={K}, max_iter={MAX_ITERATIONS} ...")
    cluster_centroids_lin = run_kmeans_on_pixels_lab_optimized(
        used_pixels, K=K, max_iter=MAX_ITERATIONS
    )
    if cluster_centroids_lin.shape[0] == 0:
        raise RuntimeError("K-means returned no centroids.")
    stepC_end = time.time()
    print(f"Stage 2 - C completed in {stepC_end - stepC_start:.2f} seconds")

    print("Stage 2 - D: Creating materials for each cluster color")
    stepD_start = time.time()
    cluster_materials = create_materials_from_centroids(cluster_centroids_lin)
    cluster_centroids_lab = convert_centroids_lin_to_lab(cluster_centroids_lin)
    stepD_end = time.time()
    print(f"Stage 2 - D completed in {stepD_end - stepD_start:.2f} seconds")

    print("Stage 2 - E: Duplicating and Remeshing the final object")
    stepE_start = time.time()
    remesh_obj = source_obj.copy()
    remesh_obj.data = source_obj.data.copy()
    remesh_obj.name = source_obj.name + "_Remeshed"
    bpy.context.collection.objects.link(remesh_obj)
    smooth_mod = remesh_obj.modifiers.new(name="SmoothRemesh", type='REMESH')
    smooth_mod.mode = 'SMOOTH'
    smooth_mod.octree_depth = 9
    smooth_mod.scale = 0.999
    smooth_mod.use_remove_disconnected = False
    bpy.context.view_layer.objects.active = remesh_obj
    bpy.ops.object.modifier_apply(modifier=smooth_mod.name)
    bpy.ops.object.select_all(action='DESELECT')
    remesh_obj.select_set(True)
    bpy.ops.export_scene.fbx(
        filepath=debug_after_remesh_fbx,
        use_selection=True,
        axis_forward='-Z',
        axis_up='Y'
    )
    print(f"Exported debug FBX after remesh to: {debug_after_remesh_fbx}")
    stepE_end = time.time()
    print(f"Stage 2 - E completed in {stepE_end - stepE_start:.2f} seconds")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    source_eval = source_obj.evaluated_get(depsgraph)
    source_mesh = source_eval.to_mesh()
    bm_source = bmesh.new()
    bm_source.from_mesh(source_mesh)
    bvh = mathutils.bvhtree.BVHTree.FromBMesh(bm_source)
    bm_source.free()
    source_eval.to_mesh_clear()

    remesh_obj.data.materials.clear()
    for (mat, _) in cluster_materials:
        remesh_obj.data.materials.append(mat)

    bm_orig = bmesh.new()
    bm_orig.from_mesh(source_obj.data)
    bm_orig.faces.ensure_lookup_table()
    bm_orig.verts.ensure_lookup_table()
    uv_layer_orig = bm_orig.loops.layers.uv.active

    # GPU-based processing for assigning material indices (Stage 2 - F)
    step_F_gpu(remesh_obj, bvh, bm_orig, uv_layer_orig, material_to_pil,
               cluster_centroids_lin, cluster_centroids_lab)
    bm_orig.free()

    print("Stage 2 - G: Planar dissolve (Limited Dissolve)")
    stepG_start = time.time()
    bpy.context.view_layer.objects.active = remesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(2.0), delimit={'MATERIAL'})
    bpy.ops.object.mode_set(mode='OBJECT')
    stepG_end = time.time()
    print(f"Stage 2 - G completed in {stepG_end - stepG_start:.2f} seconds")

    # --- Triangulation via BMesh ---
    print("Stage 2 - X: Triangulating final mesh")
    stepX_start = time.time()
    bm_triang = bmesh.new()
    bm_triang.from_mesh(remesh_obj.data)
    bmesh.ops.triangulate(bm_triang, faces=bm_triang.faces[:])
    bm_triang.to_mesh(remesh_obj.data)
    bm_triang.free()
    stepX_end = time.time()
    print(f"Stage 2 - X completed in {stepX_end - stepX_start:.2f} seconds")

    print("Stage 2 - H: Exporting final OBJ (selection-based)")
    stepH_start = time.time()
    bpy.ops.object.select_all(action='DESELECT')
    remesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = remesh_obj
    # Blender 4.x uses wm.obj_export instead of legacy export_scene.obj
    try:
        bpy.ops.wm.obj_export(
            filepath=OBJ_EXPORT_PATH,
            export_selected_objects=True,
            export_materials=True,
            forward_axis='NEGATIVE_Z',
            up_axis='Y'
        )
    except Exception as e:
        print(f"wm.obj_export failed ({e}), trying legacy export_scene.obj...")
        bpy.ops.export_scene.obj(
            filepath=OBJ_EXPORT_PATH,
            use_selection=True,
            use_materials=True,
            path_mode='AUTO',
            axis_forward='-Z',
            axis_up='Y'
        )
    stepH_end = time.time()
    print(f"Stage 2 - H completed in {stepH_end - stepH_start:.2f} seconds")

    print("Stage 2 - I: Deleting original non-remeshed model")
    original_name = source_obj.name
    bpy.data.objects.remove(source_obj, do_unlink=True)
    print(f"  Deleted original object: {original_name}")

    total_stage2_end = time.time()
    print(f"Part 2 completed successfully in {total_stage2_end - total_stage2_start:.2f} seconds.")
    print("Processing completed successfully. Exported to:", OBJ_EXPORT_PATH)

def main():
    print("=== Starting Full Processing Pipeline ===")
    overall_start = time.time()
    main_stage2()
    overall_end = time.time()
    print(f"Total processing time: {overall_end - overall_start:.2f} seconds")

if __name__ == "__main__":
    main()
