import taichi as ti
from .fields import grid_v, grid_m, levelset, cell_t, DT
from . import config as C, utils as U

GRAV = ti.Vector(C.GRAVITY)

@ti.kernel
def add_gravity():
    for I in ti.grouped(grid_v):
        if grid_m[I] > 0:
            grid_v[I] += GRAV * DT[None]

# ─── Viscosity (implicit, one Jacobi step) ─────────────────────────────────────
@ti.func
def lap_v(I, c):
    i,j,k = I
    nbr = lambda di,dj,dk: grid_v[i+di,j+dj,k+dk][c] \
        if 0<=i+di<C.RES and 0<=j+dj<C.RES and 0<=k+dk<C.RES else grid_v[I][c]
    return (nbr(1,0,0)+nbr(-1,0,0)+nbr(0,1,0)+nbr(0,-1,0)+
            nbr(0,0,1)+nbr(0,0,-1)-6*grid_v[I][c])*U.INV_DX**2

@ti.kernel
def apply_viscosity():
    for I in ti.grouped(grid_v):
        if grid_m[I]>0:
            for c in ti.static(range(3)):
                grid_v[I][c] += C.VISC * DT[None] * lap_v(I, c)

# ─── Surface tension  (Brackbill CSF) ─────────────────────────────────────────
@ti.kernel
def add_surface_tension():
    for I in ti.grouped(grid_v):
        if cell_t[I]==1:
            grad = ti.Vector.zero(ti.f32, 3)
            lap  = 0.0
            for d in ti.static(range(3)):
                inc = ti.Vector.unit(3,d)*U.DX
                φp = levelset[I + inc]
                φn = levelset[I - inc]
                grad[d] = (φp - φn)*0.5/U.DX
                lap    += (φp - 2*levelset[I] + φn)/(U.DX*U.DX)
            κ = -lap / (grad.norm()+1e-6)
            F = C.SIGMA * κ * grad.normalized()
            grid_v[I] += F * DT[None] / ti.max(grid_m[I], 1e-6)
