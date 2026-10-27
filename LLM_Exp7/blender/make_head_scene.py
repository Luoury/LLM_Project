"""Build a white-mannequin bust scene (head + neck + shoulders) for the
portrait-fusion extension of experiment 7.

    blender.exe -b --factory-startup -P make_head_scene.py -- \
        --out assets/head_bust.blend --variant bust --res 512

Variants:
  bust      head + neck + shoulders  -> used to generate the base man image
  headonly  head + neck only         -> used to generate the face that will be
                                        pasted back onto the bust
"""
import argparse
import math
import os
import sys

import bpy
import mathutils


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_clay(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    return mat


def uv_sphere(name, loc, radius, scale, mat, segments=64, rings=48):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings,
                                         radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(mat)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def cylinder(name, loc, radius, depth, mat, rot=(0, 0, 0), verts=64):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth,
                                        location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(mat)
    return obj


def box(name, loc, size, mat, bevel=0.12, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 6
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.ops.object.shade_smooth()
    obj.data.materials.append(mat)
    return obj


def area_light(name, loc, target, energy, size, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.shape = "SQUARE"
    light.data.size = size
    light.data.color = color
    direction = mathutils.Vector((target[0] - loc[0], target[1] - loc[1], target[2] - loc[2]))
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return light


def setup_camera(loc, target, lens=63.0):
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    cam.data.lens = lens
    direction = mathutils.Vector((target[0] - loc[0], target[1] - loc[1], target[2] - loc[2]))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def setup_world(color=(0.02, 0.02, 0.022)):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (*color, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0


def setup_render(res):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/head_bust.blend")
    ap.add_argument("--variant", default="bust", choices=["bust", "headonly"])
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args(argv)

    clear_scene()
    setup_world()
    clay = make_clay("clay", (0.82, 0.80, 0.78))

    # --- mannequin bust -------------------------------------------------
    uv_sphere("Head", (0.0, 0.0, 1.62), 0.62, (0.86, 0.95, 1.10), clay)
    cylinder("Neck", (0.0, 0.02, 1.16), 0.21, 0.46, clay)
    if args.variant == "bust":
        box("Shoulders", (0.0, 0.0, 0.90), (1.62, 0.62, 0.46), clay, bevel=0.16)
        box("Chest", (0.0, 0.04, 0.62), (1.30, 0.54, 0.30), clay, bevel=0.12)

    # --- lighting -------------------------------------------------------
    area_light("Key", (-2.10, -2.55, 2.75), (0, 0, 1.58), 260.0, 1.6, (1.0, 0.97, 0.93))
    area_light("Fill", (2.30, -1.90, 1.55), (0, 0, 1.50), 80.0, 2.2, (0.92, 0.95, 1.0))
    area_light("Rim", (0.50, 2.10, 2.40), (0, 0, 1.62), 150.0, 1.2, (1.0, 1.0, 1.0))

    setup_camera((0.34, -3.95, 1.78), (0.0, 0.0, 1.40), lens=63.0)
    setup_render(args.res)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print(f"[make_head_scene] saved {out} variant={args.variant}")


if __name__ == "__main__":
    main()
