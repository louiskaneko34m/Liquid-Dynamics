import argparse, taichi as ti
from . import particles as P, sim, gui, surface
from . import config as C

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vdb', action='store_true', help='export VDB')
    parser.add_argument('--frames', type=int, default=600)
    args = parser.parse_args()

    P.init_domain(); P.init_particles()
    frame = 0
    while gui.gui.running and frame < args.frames:
        for _ in range(1):          # you can expose this via CLI
            sim.substep()
        if frame % C.EXPORT_EVERY == 0:
            surface.dump_mesh(frame)
        gui.draw(frame)
        frame += 1

if __name__ == "__main__":
    run()
