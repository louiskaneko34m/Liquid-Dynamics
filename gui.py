import taichi as ti, numpy as np
from .fields import x, fx, p_count, f_count
from . import config as C

gui = ti.GUI('FLIP-T4', res=C.GUI_RES, background_color=0x112F41)

def draw(frame):
    p = x.to_numpy()[:p_count[None]]
    gui.circles(p[:,[0,2]], radius=1.5, color=0x4DC0FF)
    if f_count[None]:
        fp = fx.to_numpy()[:f_count[None]]
        gui.circles(fp[:,[0,2]], radius=1, color=0xEEEEEE)
    gui.text(f'frame {frame}', pos=(0,0.96), color=0xFFFFFF)
    gui.show()
