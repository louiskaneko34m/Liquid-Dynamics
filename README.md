# Liquid-Dynamics

Project tree

```
flip_t4/
├── __init__.py         (empty – turns the folder into a pkg)
├── config.py           # all knobs & constants
├── utils.py            # B-spline weights, timing helpers…
├── fields.py           # sparse SNode layout, global Taichi fields
├── mgpcg.py            # multigrid-preconditioned CG pressure solver
├── forces.py           # gravity, viscosity, surface-tension
├── particles.py        # P2G / G2P, foam & spray, particle advection
├── surface.py          # level-set build + marching cubes / OpenVDB
├── sim.py              # one sub-step orchestrator
├── gui.py              # real-time visualization
└── main.py             # CLI entry-point  (python -m flip_t4)
```

Install deps

```
pip install taichi rich scikit-image openvdb trimesh numpy
```
