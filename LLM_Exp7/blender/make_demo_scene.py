"""Generate the demo white-model scene for the depth-controlled pipeline.

Scene: white matte sphere / torus / beveled cube / cylinder on a ground plane,
three-point studio lighting, 63mm camera, Cycles CPU renderer.

Usage:
    blender -b --factory-startup -P blender/make_demo_scene.py -- [--out assets/demo_white_model.blend] [--res 512]
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets/demo_white_model.blend")
    p.add_argument("--res", type=int, default=512)
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def matte_material(name, color=(0.82, 0.82, 0.80, 1.0), roughness=0.55, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def shade_smooth(obj):
    for poly in obj.data.polygons:
        poly.use_smooth = True


def add_sphere(loc, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = "Sphere"
    obj.data.materials.append(mat)
    shade_smooth(obj)
    return obj


def add_torus(loc, rot, mat):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.86, minor_radius=0.30,
        major_segments=128, minor_segments=32,
        location=loc, rotation=rot,
    )
    obj = bpy.context.object
    obj.name = "Torus"
    obj.data.materials.append(mat)
    shade_smooth(obj)
    return obj


def add_cube(loc, size, rot, mat):
    bpy.ops.mesh.primitive_cube_add(size=size, location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = "BeveledCube"
    obj.data.materials.append(mat)
    bev = obj.modifiers.new(name="Bevel", type="BEVEL")
    bev.width = size * 0.055
    bev.segments = 8
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(40)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=bev.name)
    shade_smooth(obj)
    return obj


def add_cylinder(loc, rot, radius, depth, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=radius, depth=depth,
                                        location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = "Cylinder"
    obj.data.materials.append(mat)
    bev = obj.modifiers.new(name="Bevel", type="BEVEL")
    bev.width = 0.04
    bev.segments = 4
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(40)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=bev.name)
    shade_smooth(obj)
    return obj


def add_ground(size, mat):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = "Ground"
    obj.data.materials.append(mat)
    return obj


def add_area_light(name, loc, target, energy, size, color=(1, 1, 1)):
    light = bpy.data.lights.new(name=name, type="AREA")
    light.energy = energy
    light.size = size
    light.color = color
    obj = bpy.data.objects.new(name=name, object_data=light)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    direction = Vector(target) - Vector(loc)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return obj


def setup_camera(loc, target, lens=63.0):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = loc
    direction = Vector(target) - Vector(loc)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def main():
    args = parse_args()
    clear_scene()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.035, 0.036, 0.040, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world

    mat_clay = matte_material("WhiteMatte", (0.80, 0.80, 0.78, 1.0), roughness=0.52)
    mat_clay2 = matte_material("WhiteMatteWarm", (0.74, 0.73, 0.71, 1.0), roughness=0.60)

    add_ground(40.0, mat_clay2)
    add_sphere((-1.95, 0.15, 0.90), 0.90, mat_clay)
    add_torus((-0.05, 0.80, 0.97), (math.radians(74), 0.0, math.radians(16)), mat_clay)
    add_cube((1.48, 0.18, 0.70), 1.40, (math.radians(3), math.radians(5), math.radians(-24)), mat_clay)
    add_cylinder((0.55, -1.45, 0.84), (0.0, 0.0, 0.0), 0.40, 1.68, mat_clay)

    # three-point lighting: key (front-left, large softbox), fill (right, dim), rim (back-top)
    add_area_light("Key", (-4.2, -5.0, 5.4), (0, 0, 1.0), 420.0, 3.4, (1.0, 0.97, 0.93))
    add_area_light("Fill", (6.2, -3.2, 2.2), (0, 0, 1.0), 130.0, 5.0, (0.92, 0.95, 1.0))
    add_area_light("Rim", (1.5, 5.5, 4.8), (0, 0, 1.2), 280.0, 2.6, (1.0, 1.0, 1.0))

    setup_camera((5.30, -7.10, 3.60), (0.0, 0.25, 0.86), lens=63.0)

    scene.render.filepath = "//clay_preview.png"

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print("[make_demo_scene] saved", out, "res", args.res)


if __name__ == "__main__":
    main()
