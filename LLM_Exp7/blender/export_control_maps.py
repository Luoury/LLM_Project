"""Export depth / normal / clay control maps from a Blender scene (headless).

The depth map follows the guide's recipe: depth is normalised between the 2nd and
98th percentile of the camera-space depth of the *visible* mesh vertices and then
inverted (near = bright, far = dark).  Instead of relying on the compositor Mist
pass (whose node API changed in Blender 5.x), the map is produced with a Cycles
material override: Light Path > Ray Length gives the exact per-pixel camera
distance, which a Map Range node converts to the normalised inverted grey level.

Usage:
    blender -b assets/demo_white_model.blend -P blender/export_control_maps.py -- \
        --outdir output --name demo --res 512 --samples 64
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="output")
    p.add_argument("--name", default="demo")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--percentile-low", type=float, default=2.0)
    p.add_argument("--percentile-high", type=float, default=98.0)
    return p.parse_args(argv)


def visible_depth_range(scene, lo_pct, hi_pct):
    """Camera-space depth percentiles over mesh vertices that fall inside the frame."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam = scene.camera
    cam_inv = cam.matrix_world.inverted()

    lens = cam.data.lens
    sensor = cam.data.sensor_width
    tan_half = (sensor * 0.5) / lens

    depths = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        try:
            n = len(mesh.vertices)
            if n == 0:
                continue
            co = np.empty(n * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", co)
            co = co.reshape(n, 3)
            mw = np.array(obj_eval.matrix_world)
            world = co @ mw[:3, :3].T + mw[:3, 3]
        finally:
            obj_eval.to_mesh_clear()

        local = (world @ np.array(cam_inv)[:3, :3].T) + np.array(cam_inv)[:3, 3]
        z = -local[:, 2]
        ok = z > 1e-4
        if not np.any(ok):
            continue
        ndc_x = (local[ok, 0] / z[ok]) / tan_half
        ndc_y = (local[ok, 1] / z[ok]) / tan_half
        inside = (np.abs(ndc_x) <= 1.15) & (np.abs(ndc_y) <= 1.15)
        if np.any(inside):
            depths.append(z[ok][inside])

    if not depths:
        raise RuntimeError("no visible vertices found for depth normalisation")
    d = np.concatenate(depths)
    near = float(np.percentile(d, lo_pct))
    far = float(np.percentile(d, hi_pct))
    if far - near < 1e-3:
        near, far = float(d.min()), float(d.max())
    return near, far


def make_emission_material(name, color_fn):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    color_fn(nt, emit)
    return mat


def build_depth_material(near, far):
    def wire(nt, emit):
        lp = nt.nodes.new("ShaderNodeLightPath")
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = near
        mr.inputs["From Max"].default_value = far
        mr.inputs["To Min"].default_value = 1.0     # near -> white
        mr.inputs["To Max"].default_value = 0.0     # far  -> black
        mr.clamp = True
        nt.links.new(lp.outputs["Ray Length"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], emit.inputs["Color"])
    return make_emission_material("__depth_override__", wire)


def build_normal_material():
    def wire(nt, emit):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        scale = nt.nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs["Scale"].default_value = 0.5
        add = nt.nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = (0.5, 0.5, 0.5)
        nt.links.new(geo.outputs["Normal"], scale.inputs[0])
        nt.links.new(scale.outputs["Vector"], add.inputs[0])
        nt.links.new(add.outputs["Vector"], emit.inputs["Color"])
    return make_emission_material("__normal_override__", wire)


def setup_render(scene, res, samples):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"


def render_to(scene, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("[export_control_maps] wrote", path)


def main():
    args = parse_args()
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    near, far = visible_depth_range(scene, args.percentile_low, args.percentile_high)
    print(f"[export_control_maps] depth range from visible vertices: near={near:.3f} far={far:.3f}")

    os.makedirs(args.outdir, exist_ok=True)
    outdir = os.path.abspath(args.outdir)

    # save the original world so the clay pass keeps the studio background
    orig_world = scene.world
    orig_override = view_layer.material_override

    # ---- depth ----
    black_world = bpy.data.worlds.new("__black_world__")
    black_world.use_nodes = True
    bgn = next((n for n in black_world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bgn is None:
        bgn = black_world.node_tree.nodes.new("ShaderNodeBackground")
        wout = next((n for n in black_world.node_tree.nodes if n.type == "OUTPUT_WORLD"), None)
        if wout is None:
            wout = black_world.node_tree.nodes.new("ShaderNodeOutputWorld")
        black_world.node_tree.links.new(bgn.outputs["Background"], wout.inputs["Surface"])
    bgn.inputs["Color"].default_value = (0, 0, 0, 1)
    bgn.inputs["Strength"].default_value = 0.0
    scene.world = black_world
    scene.render.film_transparent = False

    setup_render(scene, args.res, 16)
    view_layer.material_override = build_depth_material(near, far)
    render_to(scene, os.path.join(outdir, f"{args.name}_depth_0001.png"))

    # ---- normal ----
    view_layer.material_override = build_normal_material()
    render_to(scene, os.path.join(outdir, f"{args.name}_normal_0001.png"))

    # ---- clay (original white matte materials + three point lighting) ----
    view_layer.material_override = None
    scene.world = orig_world
    setup_render(scene, args.res, args.samples)
    scene.cycles.use_denoising = True
    render_to(scene, os.path.join(outdir, f"{args.name}_clay.png"))

    view_layer.material_override = orig_override

    meta = {
        "scene": bpy.path.basename(bpy.data.filepath),
        "resolution": args.res,
        "clay_samples": args.samples,
        "depth_near": near,
        "depth_far": far,
        "percentiles": [args.percentile_low, args.percentile_high],
    }
    import json
    with open(os.path.join(outdir, f"{args.name}_control_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[export_control_maps] done")


if __name__ == "__main__":
    main()
