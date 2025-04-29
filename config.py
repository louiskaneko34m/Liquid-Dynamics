# All simulation knobs live here so you can tweak without touching core code.

RES             = 256          # base grid resolution (cube)
TILE            = 8            # 8×8×8 sparse bricks
PART_PER_CELL   = 8            # fluid markers
FOAM_RATIO      = 0.02         # chance to spawn foam from a primary particle
FLIP_RATIO      = 0.95
USE_APIC        = True
CFL             = 0.9
SIGMA           = 0.0728       # surface tension  (N/m  water @20 °C)
VISC            = 1.0e-3       # dynamic viscosity
GRAVITY         = (0.0, -9.81, 0.0)
ARCH            = 'cuda'       # 'cuda'|'cpu'|'vulkan' …
EXPORT_EVERY    = 25           # dump mesh every N frames
GUI_RES         = (900, 900)   # window size
