"""
Multigrid-preconditioned Conjugate Gradient on a sparse MAC grid.
Works on fields.pressure / divergence provided by fields.py
"""

import taichi as ti
from .fields import pressure, divergence, cell_t
from . import config as C, utils as U

L_MAX = 3                    # levels 0(h),1(2h),2(4h)
resL  = [C.RES//(2**l) for l in range(L_MAX)]

# allocate per-level scratch
pL = [ti.field(ti.f32, shape=(r,r,r)) for r in resL]
rL = [ti.field(ti.f32, shape=(r,r,r)) for r in resL]

@ti.func
def lap(p, I):
    i,j,k = I
    R = p.shape[0]
    c = p[I]
    l = p[i-1,j,k] if i>0 else c
    r = p[i+1,j,k] if i<R-1 else c
    d = p[i,j-1,k] if j>0 else c
    u = p[i,j+1,k] if j<R-1 else c
    b = p[i,j,k-1] if k>0 else c
    f = p[i,j,k+1] if k<R-1 else c
    return (l+r+d+u+b+f - 6*c) * U.INV_DX**2

@ti.kernel
def restrict_fine_to_coarse(fine: ti.template(), coarse: ti.template()):
    for I in ti.grouped(coarse):
        i2,j2,k2 = I*2
        s = (fine[i2,j2,k2]+fine[i2+1,j2,k2]+fine[i2,j2+1,k2]+fine[i2,j2,k2+1]+
             fine[i2+1,j2+1,k2]+fine[i2+1,j2,k2+1]+fine[i2,j2+1,k2+1]+fine[i2+1,j2+1,k2+1])
        coarse[I] = s * 0.125

@ti.kernel
def prolongate(coarse: ti.template(), fine: ti.template()):
    for I in ti.grouped(fine):
        fine[I] = coarse[I//2]

@ti.kernel
def jacobi_relax(x: ti.template(), b: ti.template()):
    for I in ti.grouped(x):
        x[I] = (b[I] - lap(x, I)*0.5) * (1/3)  # ω≈2/3

def v_cycle(b, x, tmp, lvl):
    if lvl == L_MAX-1:        # coarsest → damp a few times
        for _ in range(16):
            jacobi_relax(x, b); ti.sync()
        return
    # pre-smooth
    for _ in range(3):
        jacobi_relax(x, b); ti.sync()
    # residual
    restrict_fine_to_coarse(b - lap(x, I=None), rL[lvl+1]); ti.sync()
    pL[lvl+1].fill(0)
    v_cycle(rL[lvl+1], pL[lvl+1], tmp, lvl+1)
    # prolongate + correct
    prolongate(pL[lvl+1], x); ti.sync()
    # post-smooth
    for _ in range(3):
        jacobi_relax(x, b); ti.sync()

@U.chrono("MGPCG pressure")
def solve():
    pressure.fill(0.0)
    v_cycle(divergence, pressure, rL[0], 0)
