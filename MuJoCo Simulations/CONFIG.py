# central simulation parameters, applied to every case unless the case's "model" block in
# settings.json (or an --override_simulation file) overrides them
# contains solver/integrator settings, youngs modulus, elasticity and damping

DT = 0.002  # seconds
SOLVER = "Newton" # default solver for flex
INTEGRATOR = "discrete" # default integrator for flex
SOLVER_ITERATIONS = 500 # default solver iterations for flex (very high for CG, but Newton converges in fewer iterations)
GRAVITY = [0, 0, 0] # default gravity for flex

YOUNG_MODULUS = 40e3 # default Young's modulus for flex (Pa; was 55e3); in the run folder names as _E<kPa>kPa
ELASTICITY = 0.485 # default Poisson's ratio for flex
DAMPING = 0.002 # default damping for flex (Rayleigh stiffness coefficient, s)
SPEED = 0.001 # default indentation speed, m/s (= 1 mm/s)

# flex contact: the contact time constant (solref[0]) is always CONTACT_TIMECONST_FACTOR x the
# timestep (shorter: contacts overshoot and chatter; longer: softer contact), and the flex contact
# gets priority so its solref/solimp are used instead of being averaged with the tool's defaults
CONTACT_TIMECONST_FACTOR = 2
CONTACT_PRIORITY = 1
# flex contact impedance (solimp), the same for every case (overrides the xml): [dmin, dmax, width (m), midpoint,
# power]. Impedance rises from dmin at zero penetration to dmax at a penetration of width (1 mm); higher
# impedance = stiffer contact (less of the constraint force is softened)
CONTACT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2]

# paths, relative to the repository root
REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
SETTINGS_FILE = REPO_ROOT / "src" / "settings.json"
OUTPUT_DIR = REPO_ROOT / "output"
MESH_STUDY_DIR = OUTPUT_DIR / "mesh_study"   # runs with --override_xml {"model": {"mesh_count": [...]}}, mesh_study.py
# another folder (e.g. for a run with another MuJoCo version): environment variable MESH_STUDY_NAME=mesh_study_mujoco4.0.0
MESH_STUDY_DIR = OUTPUT_DIR / __import__("os").environ.get("MESH_STUDY_NAME", MESH_STUDY_DIR.name)
CONTACT_STUDY_DIR = OUTPUT_DIR / "contact_study"   # contact_study.py: same runs with other flex solimp
CONTACT_STUDY = {"default": [0.9, 0.95, 0.001, 0.5, 2],      # MuJoCo's default
                 "current": [0.95, 0.99, 0.001, 0.5, 2],     # CONTACT_SOLIMP
                 "stiff": [0.99, 0.999, 0.001, 0.5, 2],      # higher impedance
                 "narrow": [0.95, 0.99, 0.0001, 0.5, 2],     # full impedance after 0.1 mm penetration
                 "wide": [0.95, 0.99, 0.005, 0.5, 2]}        # full impedance after 5 mm penetration
CONTACT_STUDY_CASES = ["case1", "case2"]
MESH_STUDY = {"case1": [[11, 11, 5], [15, 15, 6], [21, 21, 8], [29, 29, 11], [36, 36, 13]],   # ~7, 5, 3.5, 2.5, 2 mm
              "case2": [[11, 11, 5], [15, 15, 6], [21, 21, 8], [29, 29, 11], [36, 36, 13]]}
#
#  COMSOL indentation simulations of the same cases as MuJoCo: every COMSOL export is read from this one folder
# (validation_analysis.load_comsol), per case and speed v:
#   <case>_force_<v>mm_s.xlsx              displacement mm / force N (at 1 mm/s also <case>_Force.xlsx)
#   <case>_vol strain energy_<v>mm_s.xlsx  time s / elastic strain energy J, volume integral (same scale as MuJoCo),
#                                          on the time grid of the force export of the same run
#   <case>_power_<v>mm_s.xlsx              time s / -solid.nI W, optional (at 1 mm/s also <case>_Power.xlsx)
# a file whose header shows another quantity (e.g. a -solid.nI export named force) is reported and skipped
# COMSOL_DIR = # SETFILEPATH

# common indentation (m) at which all sources (MuJoCo SVK / SNH, COMSOL) and all speeds of a case are
# compared: the smallest maximum depth over them (case1: COMSOL 1 mm/s; case2, case3: MuJoCo). Cases not listed
# fall back on the smallest maximum depth of the runs being compared
D_REF = {"case1": 0.00850, "case2": 0.00441, "case3": 0.00845}

# viewer / video
SYNC_EVERY = 100          # redraw the passive viewer every N steps (rendering is slow)
VIDEO_FPS = 30
VIDEO_SIZE = (480, 640)  # height, width
VIDEO_MAX_FRAMES = 400   # frames are kept in memory until saved (gif): fewer frames for long runs
VIDEO_FORCE_SCALE = 0.1  # length of the contact force arrows in the video, m per N (~0.04 N per contact = 4 mm)
SNAPSHOT_SIZE = (960, 1280)   # height, width of each view in the snapshots (--save_snapshots)
SNAPSHOT_VIEWS = [(45, -20), (90, 0)]   # (azimuth, elevation) in degrees: oblique view, side view
SAVE_EVERY = 10         # store every N-th step in the saved results
