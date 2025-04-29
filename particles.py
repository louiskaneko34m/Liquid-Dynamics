import taichi as ti, math
from . import config as C, utils as U
from .fields import *
from .utils  import w_bspline

# ─── Init (box of fluid) ──────────────────────────────────────────────────────
@ti.kernel
def init_domain():
    # solids: floor + walls
    for I in ti.grouped(cell_t):
        i,j,k = I
        cell_t[I] = 0
        if i==0 or i==C.RES-1 or k==0 or k==C.RES-1 or j==0:
            cell_t[I] = 2
    # fluid block
    for I in ti.grouped(cell_t):
        p = (ti.Vector(I, dt=ti.f32)+0.5) * U.DX
        if (0.15<p.x<0.85) and (0.15<p.y<0.65) and (0.15<p.z<0.85):
            cell_t[I] = 1

@ti.kernel
def init_particles():
    p_count[None] = 0
    for I in ti.grouped(cell_t):
        if cell_t[I]==1:
            for _ in range(C.PART_PER_CELL):
                pid = ti.atomic_add(p_count[None], 1)
                offs = ti.Vector([ti.random(),ti.random(),ti.random()])
                x[pid]  = (ti.Vector(I)+offs)*U.DX
                v_p[pid]= ti.Vector.zero(ti.f32,3)
                C_apic[pid] = ti.Matrix.zero(ti.f32,3,3)

# ─── P2G / G2P  (quadratic B-spline) ─────────────────────────────────────────
@ti.kernel
def p2g():
    for pid in range(p_count[None]):
        base = (x[pid]*U.INV_DX-0.5).cast(int)
        fx   = x[pid]*U.INV_DX - base.cast(ti.f32)
        for i,j,k in ti.static(ti.ndrange(2,2,2)):
            weight = (w_bspline(fx.x-i+0.5)*
                      w_bspline(fx.y-j+0.5)*
                      w_bspline(fx.z-k+0.5))
            node = base + ti.Vector([i,j,k])
            if (0<=node[0]<C.RES and 0<=node[1]<C.RES and 0<=node[2]<C.RES):
                vel = v_p[pid]
                if C.USE_APIC:
                    rel = (ti.Vector([i,j,k])-fx+0.5)*U.DX
                    vel += C_apic[pid]@rel
                m = weight
                ti.atomic_add(grid_m[node], m)
                ti.atomic_add(grid_v[node], m*vel)
    for I in ti.grouped(grid_m):
        if grid_m[I]>0:
            grid_v[I] /= grid_m[I]

@ti.kernel
def g2p():
    for pid in range(p_count[None]):
        base = (x[pid]*U.INV_DX-0.5).cast(int)
        fx   = x[pid]*U.INV_DX - base.cast(ti.f32)
        new_v= ti.Vector.zero(ti.f32,3)
        new_C= ti.Matrix.zero(ti.f32,3,3)
        for i,j,k in ti.static(ti.ndrange(2,2,2)):
            weight = (w_bspline(fx.x-i+0.5)*
                      w_bspline(fx.y-j+0.5)*
                      w_bspline(fx.z-k+0.5))
            node = base + ti.Vector([i,j,k])
            if 0<=node[0]<C.RES and 0<=node[1]<C.RES and 0<=node[2]<C.RES:
                node_v = grid_v[node]
                new_v += weight*node_v
                if C.USE_APIC:
                    rel = (ti.Vector([i,j,k])-fx+0.5)*U.DX
                    new_C += 4*weight*ti.outer_product(node_v, rel)*U.INV_DX
        dv = new_v - v_p[pid]
        v_p[pid] = v_p[pid] + C.FLIP_RATIO*dv + (1-C.FLIP_RATIO)*(new_v-dv)
        if C.USE_APIC:
            C_apic[pid] = new_C

# ─── Particle advection + boundary bounce ─────────────────────────────────────
@ti.kernel
def advect_particles():
    for pid in range(p_count[None]):
        x[pid] += v_p[pid] * DT[None]
        for d in ti.static(range(3)):
            if x[pid][d] < U.DX:
                x[pid][d] = U.DX;   v_p[pid][d] *= -0.5
            if x[pid][d] > 1-U.DX:
                x[pid][d] = 1-U.DX; v_p[pid][d] *= -0.5

# ─── Foam / spray ─────────────────────────────────────────────────────────────
@ti.kernel
def spawn_foam():
    for pid in range(p_count[None]):
        if ti.random()<C.FOAM_RATIO:
            fid = ti.atomic_add(f_count[None], 1)
            fx[fid] = x[pid]
            fv[fid] = v_p[pid] + ti.Vector([ti.randn(),ti.randn(),ti.randn()])*0.5
            flife[fid] = 1.0

@ti.kernel
def step_foam():
    for i in range(f_count[None]):
        fv[i] += ti.Vector(C.GRAVITY) * DT[None]
        fx[i] += fv[i] * DT[None]
        flife[i] -= DT[None]*0.4
