"""
Single sub-step orchestrator.
"""
from . import utils as U, forces as F, mgpcg, particles as P
from .fields import *
import taichi as ti, math
from . import config as C

@ti.kernel
def compute_dt():
    vmax = 0.0
    for I in ti.grouped(grid_v):
        ti.atomic_max(vmax, grid_v[I].norm())
    cap = (U.DX**3*1000.0) / C.SIGMA
    DT[None] = min(C.CFL*U.DX/(vmax+1e-6), 0.25*cap, 4e-3)

def substep():
    compute_dt()
    P.p2g()
    F.add_gravity(); F.apply_viscosity()
    # (center->faces / divergence) fused in mgpcg for brevity
    mgpcg.solve()
    F.add_surface_tension()
    P.g2p(); P.advect_particles()
    P.spawn_foam(); P.step_foam()
