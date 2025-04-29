import taichi as ti
from   . import config as C

ti.init(arch=getattr(ti, C.ARCH), packed=True)   # e.g. ti.cuda
DX = 1.0 / C.RES

# ─── Sparse grid layout ────────────────────────────────────────────────────────
root = ti.root
block = root.dense(ti.ijk, C.RES // C.TILE)
leaf  = block.dynamic(ti.ijk, 4096, chunk_size=C.TILE**3)

vec3f = lambda: ti.Vector.field(3, ti.f32)
mat3f = lambda: ti.Matrix.field(3, 3, ti.f32)

grid_v   = vec3f();  grid_m  = ti.field(ti.f32)
pressure = ti.field(ti.f32); divergence = ti.field(ti.f32)
cell_t   = ti.field(ti.i8)    # 0 air, 1 fluid, 2 solid
levelset = ti.field(ti.f32)   # signed distance φ

leaf.place(grid_v, grid_m, pressure, divergence, cell_t, levelset)

# staggered faces (dense, cheap)
u = ti.field(ti.f32, shape=(C.RES+1, C.RES, C.RES))
v = ti.field(ti.f32, shape=(C.RES, C.RES+1, C.RES))
w = ti.field(ti.f32, shape=(C.RES, C.RES, C.RES+1))

# particles (fluid + foam)
MAX_P = (C.RES**3)*C.PART_PER_CELL // 3
x   = vec3f();   v_p = vec3f();  C_apic = mat3f()
root.dense(ti.i, MAX_P).place(x, v_p, C_apic)
p_count = ti.field(ti.i32, shape=())

FOAM_MAX = 4_000_000
fx = vec3f(); fv = vec3f(); flife = ti.field(ti.f32)
root.dense(ti.i, FOAM_MAX).place(fx, fv, flife)
f_count = ti.field(ti.i32, shape=())

# adaptive dt
DT = ti.field(ti.f32, shape=())
