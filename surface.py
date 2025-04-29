"""
Build signed-distance field from markers and export mesh.
"""
import numpy as np, skimage.measure, trimesh, os, taichi as ti
from . import utils as U, config as C
from .fields import x, p_count, levelset

@ti.kernel
def clear_levelset():
    for I in ti.grouped(levelset):
        levelset[I] = 1.0

@ti.kernel
def accumulate_sdf():
    for pid in range(p_count[None]):
        P = x[pid]
        base = (P*U.INV_DX).cast(int)
        for i,j,k in ti.static(ti.ndrange((-1,2),(-1,2),(-1,2))):
            node = base + ti.Vector([i,j,k])
            if 0<=node[0]<C.RES and 0<=node[1]<C.RES and 0<=node[2]<C.RES:
                cell = (ti.Vector(node)+0.5)*U.DX
                dist = (cell-P).norm()
                levelset[node] = ti.min(levelset[node], dist)

def dump_mesh(frame):
    clear_levelset(); accumulate_sdf(); ti.sync()
    sdf = levelset.to_numpy()
    verts, faces, _, _ = skimage.measure.marching_cubes(sdf, level=U.DX*1.3)
    verts *= U.DX
    mesh = trimesh.Trimesh(verts, faces)
    os.makedirs('meshes', exist_ok=True)
    mesh.export(f'meshes/frame_{frame:04d}.ply')
    print(f'[dump] mesh frame {frame} -> {len(verts)} verts')
