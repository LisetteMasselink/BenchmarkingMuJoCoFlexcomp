# flexcompvalidation

Validation of MuJoCo's 3D flex (`flexcomp`) as a model for soft tissue. A rigid tool presses into a pinned soft
body, and the force, strain energy and energy flow are compared with COMSOL finite-element simulations and, for
case 1, with Instron indentation tests on Ecoflex 00-10.

For detailed instructions (all options, analysis outputs, what the numbers mean, known issues), see
[INSTRUCTIONS.md](INSTRUCTIONS.md).

## Validation cases

| Case | Model | Load case | Speeds (mm/s) |
|------|-------|-----------|---------------|
| `case1` | [model1.xml](src/models/model1.xml) | Sphere (R = 10 mm) indents a 70 × 70 × 25 mm block by 9 mm | 1, 2, 5 |
| `case2` | [model2.xml](src/models/model2.xml) | Block compressed 5 mm between two plates | 1, 2 |
| `case3` | [model3.xml](src/models/model3.xml) | Exoskeleton plate presses 10 mm on a hollow thigh pinned to the femur | 1 |

Each case runs with two hyperelastic materials: Saint Venant–Kirchhoff (SVK, `--elastic3d 0`) and stable
neo-Hookean (SNH, `--elastic3d 1`). Every run loads and then unloads at constant speed.

Main settings ([src/CONFIG.py](src/CONFIG.py)): E = 40 kPa, ν = 0.485, density 1040 kg/m³, timestep 2 ms, discrete
integrator, Newton solver, flex damping 0.002 s, contact time constant 2 × timestep.

## Quick start

```bash
python -m venv updatedmujoco
updatedmujoco/Scripts/python.exe -m pip install -r requirements.txt

PY=updatedmujoco/Scripts/python.exe
$PY src/main.py --case case1 --elastic3d 1 --no_viewer   # one run -> output/<case>_<material>_speed..._damping.../
bash rerun_all.sh                                        # all 12 validation runs (about 5 h)
$PY src/comsol_comparison.py                             # all runs against COMSOL and Instron
```

The COMSOL exports aren't in the repository. Set their location in `COMSOL_DIR` in
[src/CONFIG.py](src/CONFIG.py), and check `INSTRON_DIR` on another computer.

## MuJoCo version

Simulation results change between MuJoCo versions, so the version matters:

| Environment | MuJoCo | Use |
|-------------|--------|-----|
| `venv/` | 3.16.0 | Earlier results (CG solver); version comparison |
| `updatedmujoco/` | 4.0.0 | Current results. Case 3 with Newton doesn't run on 3.16 |
| `updatedmujoco_401/` | 4.0.1 | Pinned in [requirements.txt](requirements.txt); fixes a sparse Cholesky bug in 4.0.0's Newton solver |

The environments themselves aren't in the repository.

## Repository layout

```
src/                      current code
  main.py                 run one case (run.py does the same)
  simulation.py           simulation class for all cases
  measures.py             per-step measures: forces, strain energy, power, energy balance
  CONFIG.py, settings.json  defaults and per-case settings
  models/                 the case models (xml) and meshes
models/, setupfiles/      earlier models, meshes and setup scripts (pinning the thigh to the femur)
output/                   results (not in git)
```
