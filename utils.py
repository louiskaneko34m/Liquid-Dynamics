import taichi as ti, time, functools
from . import config as C

DX       = 1.0 / C.RES
INV_DX   = 1.0 / DX

# Quadratic B-spline weight & derivative ---------------------------------------
@ti.func
def w_bspline(x):
    ax = ti.abs(x)
    if ax < 1.0:
        return 0.5*ax**3 - ax**2 + 2/3
    elif ax < 2.0:
        return (-1/6)*ax**3 + ax**2 - 2*ax + 4/3
    return 0.0

@ti.func
def dw_bspline(x):
    s  = ti.sign(x)
    ax = ti.abs(x)
    if ax < 1.0:
        return s * (1.5*ax**2 - 2*ax)
    elif ax < 2.0:
        return s * (-0.5*ax**2 + 2*ax - 2)
    return 0.0

# micro decorator to time kernels ----------------------------------------------
def chrono(msg):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            ti.sync(); t0 = time.perf_counter()
            out = fn(*a, **k)
            ti.sync(); dt = (time.perf_counter()-t0)*1e3
            print(f'{msg}: {dt:5.1f} ms')
            return out
        return wrap
    return deco
