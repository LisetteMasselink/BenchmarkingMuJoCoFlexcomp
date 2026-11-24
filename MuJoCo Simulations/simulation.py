"""One simulation class for all validation cases (src/settings.json): a position-controlled rigid tool
(indentor/plates/exo represented as a plate) presses on a flex (soft tissue) that is pinned to a base (plate / bone (femur)).

Model parameters: CONFIG.py defaults, overridden per case by the "model" block in settings.json and by an
--override_simulation json. 
Motion is a ramp: settle, approach until first contact, press indent_depth at speed, hold, unload at speed
Measures per step: measures.ContactMeasures (contact force and power, energies, force to the base, ...).
"""
import contextlib
import math
import time as _time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import CONFIG
from measures import ContactMeasures
from progress_display import SimulationProgress
from utils import load_case_settings_model, load_case_settings_simulation, save_data
from xmlparser import set_grid_density


def _model_defaults():
    "all the model defaults from CONFIG.py, for the xml reader to apply to the mjSpec before compilation"
    return {"timestep": CONFIG.DT, "solver": CONFIG.SOLVER, "integrator": CONFIG.INTEGRATOR,
            "iterations": CONFIG.SOLVER_ITERATIONS, "gravity": list(CONFIG.GRAVITY),
            "young": CONFIG.YOUNG_MODULUS, "poisson": CONFIG.ELASTICITY, "damping": CONFIG.DAMPING, "speed": CONFIG.SPEED,
            "contact_timeconst_factor": CONFIG.CONTACT_TIMECONST_FACTOR, "contact_priority": CONFIG.CONTACT_PRIORITY,
            "contact_solimp": list(CONFIG.CONTACT_SOLIMP),
            "study": None,   # sensitivity study: runs saved in OUTPUT_DIR/<study>/ (see contact_study.py)
            "mesh_count": None}  # None: the flexcomp count of the xml; [nx, ny, nz]: re-mesh a grid flex

class Simulation(object):
    def __init__(self,
                 case='case1',
                 elastic3d=0,
                 save_video=False,
                 save_data=True,
                 override_simulation=False,
                 override_xml=False, # override some settings in the xml file 
                 xml=None, # give another xml path as input 
                 debugger=False,
                 viewer=False,
                 save_snapshots=False): # png of the scene at first contact and at max indentation
        "initialize simulation with the primary settings"
        self.case = case # case 1, 2, or 3 
        self.elastic3d = elastic3d #StVK (0) or SNH (1) 
        self.save_video, self.save_data_flag, self.debugger, self.viewer = save_video, save_data, debugger, viewer
        self.save_snapshots = save_snapshots
        self.settings_model = load_case_settings_model(CONFIG.SETTINGS_FILE, case, override_xml)
        self.settings_simulations = load_case_settings_simulation(CONFIG.SETTINGS_FILE, case, override_simulation)
        self.settings = {**self.settings_model, **self.settings_simulations}
        self.overrides = {"override_xml": str(override_xml) if override_xml else None,
                          "override_simulation": str(override_simulation) if override_simulation else None}

        self.protocol = self.settings["protocol"] 
        xml = Path(xml or self.settings["xml"])
        self.xml = xml if xml.is_absolute() else CONFIG.REPO_ROOT / xml
        self.params = _model_defaults() # load the default model parameters
        self.params.update(self.settings.get("model") or {})
        self.material = "SVK" if elastic3d == 0 else "SNH"
        
        # young and mesh in the folder name, so runs with other model settings do not overwrite each other
        mesh = "x".join(str(n) for n in self.params["mesh_count"]) if self.params["mesh_count"] else "xmlmesh"
        self.outdir = CONFIG.OUTPUT_DIR / "MuJoCo401" / f"{case}_{self.material}_speed{self.protocol['speed']}_E{self.params['young'] / 1e3:g}kPa_damping{self.params['damping']:g}"
        if self.params["mesh_count"]:   # mesh sensitivity runs: own folder, mesh in the name
            self.outdir = CONFIG.MESH_STUDY_DIR / f"{self.outdir.name}_mesh{mesh}"
        if self.params.get("study"):    # sensitivity study runs ("study" in the override json): own folder
            solimp = "_".join(f"{v:g}" for v in self.params["contact_solimp"])
            self.outdir = CONFIG.OUTPUT_DIR / self.params["study"] / f"{self.outdir.name}_solimp{solimp}"
        self.read_xml()
        self.setup()

    # ----------------------------------------------------------------------------------------- model
    def read_xml(self):
        """Load the xml through mjSpec, apply the model parameters, compile.
        The mesh density is changed in the xml text (the flexcomp is already expanded into vertices and pins
        in the mjSpec); the material and solver settings on the mjSpec."""
        if self.params["mesh_count"]:
            spec = mujoco.MjSpec.from_string(set_grid_density(self.xml, self.settings["flex"], self.params["mesh_count"]))
        else:
            spec = mujoco.MjSpec.from_file(str(self.xml))

        # set simulation parameters 
        spec.option.timestep = self.params["timestep"]
        spec.option.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + self.params["integrator"].upper())
        spec.option.solver = getattr(mujoco.mjtSolver, "mjSOL_" + self.params["solver"].upper())
        spec.option.iterations = self.params["iterations"]
        spec.option.gravity = self.params["gravity"]

        # set flex parameters 
        flexes = [f for f in spec.flexes if f.name == self.settings["flex"]]
        if not flexes:
            raise KeyError(f"flex '{self.settings['flex']}' not found in {self.xml}")
        flex = flexes[0]
        flex.elastic3d = self.elastic3d
        flex.young, flex.poisson, flex.damping = self.params["young"], self.params["poisson"], self.params["damping"]
        # contact time constant = factor x timestep; priority so these settings are not averaged
        # (solmix) with the tool's default solref
        solref = list(flex.solref)
        solref[0] = self.params["contact_timeconst_factor"] * self.params["timestep"]
        flex.solref = solref
        flex.solimp = self.params["contact_solimp"]   # same impedance in every case, not the xml's
        flex.priority = self.params["contact_priority"]
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

    def setup(self):
        """Find the flex, tool, actuator and base; gravity compensation of the tool; contact surface."""
        name2id = lambda kind, name: mujoco.mj_name2id(self.model, kind, name)
        self.flex_id = name2id(mujoco.mjtObj.mjOBJ_FLEX, self.settings["flex"])
        if "geoms" in self.settings['tool']:
            self.tool_geoms = [name2id(mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.settings['tool']["geoms"]]
        else:   # all geoms on the tool body and its child bodies (e.g. split plates)
            body = name2id(mujoco.mjtObj.mjOBJ_BODY, self.settings['tool']["body"])
            self.tool_geoms = [g for g in range(self.model.ngeom)
                               if self.model.geom_bodyid[g] == body or self.model.body_parentid[self.model.geom_bodyid[g]] == body]
        joint = name2id(mujoco.mjtObj.mjOBJ_JOINT, self.settings['tool']["joint"])
        self.actuator = name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, self.settings['tool']["actuator"])
        self.base_geoms = [name2id(mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.settings['base']["geoms"]]
        ids = [self.flex_id, joint, self.actuator, *self.tool_geoms, *self.base_geoms]
        
        if min(ids) < 0 or not self.tool_geoms: # -1 does mean there is no object 
            raise KeyError(f"a flex/tool/actuator/base name in settings.json was not found in {self.xml}")
        self.tool_dof, self.tool_qadr = self.model.jnt_dofadr[joint], self.model.jnt_qposadr[joint]
        
        tool_body = self.model.jnt_bodyid[joint]
        # carry the tool's own weight (as a test rig would), so it does not sag below its target and
        # the load cell does not read its weight on top of the contact force (should not be necessary as in the current config, grav = 0)
        self.data.qfrc_applied[self.tool_dof] = -self.model.body_subtreemass[tool_body] * self.model.opt.gravity[2]
        mujoco.mj_forward(self.model, self.data) # map state to acceleration 

        adr, num = self.model.flex_vertadr[self.flex_id], self.model.flex_vertnum[self.flex_id] # find vertex addresses and numbers 
        self.x_rest = self.data.flexvert_xpos[adr:adr + num].copy() # identify all the vertices 
        self.surface_vertices = self.surface_area = None
        if self.settings.get("surface") == "top":   # flat top layer of a grid flex: area per vertex from the grid spacing
            x0 = self.x_rest
            top = np.where(np.isclose(x0[:, 2], x0[:, 2].max()))[0]
            xs = np.unique(np.round(x0[top, 0], 9))
            spacing = np.min(np.diff(xs))
            edge_x = np.isclose(np.abs(x0[top, 0]), np.abs(x0[:, 0]).max())
            edge_y = np.isclose(np.abs(x0[top, 1]), np.abs(x0[:, 1]).max())
            self.surface_vertices = top
            self.surface_area = spacing ** 2 * np.where(edge_x, 0.5, 1.0) * np.where(edge_y, 0.5, 1.0)
        tool_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.tool_geoms]
        print(f"{self.case} ({self.settings.get('description', '')})\n  xml {self.xml.name}, material {self.material}, "
              f"dt {self.model.opt.timestep}, damping {self.params['damping']:g}, flex {num} vertices, "
              f"tool {len(tool_names)} geom(s), base {self.settings['base']['name']}")

    # -------------------------------------------------------------------------------------- protocol
    def _after_contact_duration(self):
        if self.protocol["type"] == "ramp":
            return 2 * self.protocol["indent_depth"] / self.protocol["speed"] + self.protocol.get("hold_time", 0.0)
        raise ValueError(f"unknown protocol type {self.protocol['type']}")

    def depth_after_contact(self, t_in):
        """Commanded indentation (m) a time t_in after first contact, and a phase label."""
        if self.protocol["type"] == "ramp":
            ramp = self.protocol["indent_depth"] / self.protocol["speed"]
            hold = self.protocol.get("hold_time", 0.0)
            if t_in < ramp:
                return self.protocol["indent_depth"] * t_in / ramp, "indent"
            if t_in < ramp + hold:
                return self.protocol["indent_depth"], "hold"
            return self.protocol["indent_depth"] * max(1.0 - (t_in - ramp - hold) / ramp, 0.0), "unload"
        cycle = self.protocol["n_cycles"] / self.protocol["frequency"]
        if t_in < self.protocol["preload_time"]:
            return self.protocol["preload_depth"] * t_in / self.protocol["preload_time"], "preload"
        if t_in < self.protocol["preload_time"] + cycle:
            tc = t_in - self.protocol["preload_time"]
            k = int(tc * self.protocol["frequency"]) + 1
            return self.protocol["preload_depth"] + self.protocol["amplitude"] * np.sin(2 * np.pi * self.protocol["frequency"] * tc), f"cycle {k}"
        frac = max(1.0 - (t_in - self.protocol["preload_time"] - cycle) / self.protocol["release_time"], 0.0)
        return self.protocol["preload_depth"] * frac, "release"

    def target(self, t):
        """Commanded tool joint position at time t, and a phase label."""
        if t < self.protocol["settle_time"]:
            return 0.0, "settle"
        if self.contact_time is None:
            return -self.protocol["speed"] * (t - self.protocol["settle_time"]), "approach"
        depth, phase = self.depth_after_contact(t - self.contact_time)
        return self.contact_target - depth, phase

    # ------------------------------------------------------------------------------------------- run
    def run(self):
        dt = self.model.opt.timestep
        max_approach = self.protocol.get("max_approach_time", 20.0)
        after = self._after_contact_duration()
        nsteps = int(np.ceil((self.protocol["settle_time"] + max_approach + after) / dt)) + 1
        self.contact_time = self.contact_target = self.contact_q = None
        self.measures = ContactMeasures(self.model, self.data, self.flex_id, self.tool_geoms, self.tool_dof, nsteps,
                                        base_geom_ids=self.base_geoms, surface_vertices=self.surface_vertices,
                                        surface_area=self.surface_area,
                                        material=(self.params["young"], self.params["poisson"], self.elastic3d))
        log = {k: np.zeros(nsteps) for k in ("time", "tool_q", "q_target", "actuator_force", "step_time",
                                              "solver_niter", "ncon_total")}
        if self.debugger:
            log["max_vertex_speed"] = np.zeros(nsteps)
            log["ncon_per_tool_geom"] = np.zeros((nsteps, len(self.tool_geoms)))
        flex_dofs = self.measures.dofs

        frames = []
        renderer = None
        frame_every = 1
        # snapshots: the full state at first contact and at the deepest tool position, rendered after the run
        self._snap_states = {}
        self._snap_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        snap_buf = np.zeros(mujoco.mj_stateSize(self.model, self._snap_spec)) if self.save_snapshots else None
        deepest_q = np.inf
        if self.save_video:
            renderer, camera, vis = self._renderer(*CONFIG.VIDEO_SIZE)
            # one frame per 1/VIDEO_FPS s of simulated time, but at most ~VIDEO_MAX_FRAMES frames
            expected_steps = (self.protocol["settle_time"] + after + 1.0) / dt
            frame_every = max(1, round(1 / (CONFIG.VIDEO_FPS * dt)), np.ceil(expected_steps / CONFIG.VIDEO_MAX_FRAMES))

        progress = SimulationProgress(self.protocol["settle_time"] + after + 0.1, dt)


        if self.viewer: 
            viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data) 
        else: 
            viewer_ctx = contextlib.nullcontext() # default to do nothing 

        wall_start = _time.perf_counter()
        i = 0
        with viewer_ctx as viewer:
            if viewer is not None:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            while i < nsteps and (viewer is None or viewer.is_running()):
                log["time"][i] = self.data.time

                # set the target 
                self.data.ctrl[self.actuator], _ = self.target(self.data.time)
                log["q_target"][i], phase = self.target(self.data.time)
                
                # start at t0
                t0 = _time.perf_counter()
                
                # take a step 
                mujoco.mj_step(self.model, self.data)
                log["step_time"][i] = _time.perf_counter() - t0

                # record displacement, the actuator solve, the solver iterations and number of contacts 
                self.measures.record(i, self.data.qpos[self.tool_qadr])
                log["tool_q"][i] = self.data.qpos[self.tool_qadr]
                log["actuator_force"][i] = self.data.actuator_force[self.actuator]
                log["solver_niter"][i] = self.data.solver_niter[0]
                log["ncon_total"][i] = self.data.ncon

                # if the debugger is one, calculate the vertex speed 
                if self.debugger:
                    log["max_vertex_speed"][i] = np.abs(self.data.qvel[flex_dofs]).max()
                    for k, g in enumerate(self.tool_geoms):
                        log["ncon_per_tool_geom"][i, k] = sum(1 for c in self.data.contact[:self.data.ncon] if g in c.geom)

                # 
                if self.contact_time is None and self.data.time >= self.protocol["settle_time"] and self.measures.values["ncon_tool"][i] > 0:
                    self.contact_target, _ = self.target(self.data.time)     # continue the profile from here
                    self.contact_time, self.contact_q = self.data.time, self.data.qpos[self.tool_qadr]
                    print(f"\n  first contact at t = {self.contact_time:.3f} s, tool at {self.contact_q * 1e3:.3f} mm")
                    if snap_buf is not None:
                        mujoco.mj_getState(self.model, self.data, snap_buf, self._snap_spec)
                        self._snap_states["first_contact"] = snap_buf.copy()
                # deepest tool position after first contact (max indentation)
                if snap_buf is not None and self.contact_time is not None and log["tool_q"][i] < deepest_q:
                    deepest_q = log["tool_q"][i]
                    mujoco.mj_getState(self.model, self.data, snap_buf, self._snap_spec)
                    self._snap_states["max_indentation"] = snap_buf.copy()

                if viewer is not None and i % CONFIG.SYNC_EVERY == 0: # if we are vieuwing 
                    viewer.sync()
                if renderer is not None and i % frame_every == 0: # if we are taking a video
                    renderer.update_scene(self.data, camera, vis)
                    frames.append(renderer.render().copy())
                if i % 50 == 0:
                    progress.update(self.data.time, i, phase, log["step_time"][i])
                i += 1

                if self.contact_time is None and self.data.time > self.protocol["settle_time"] + max_approach:
                    print(f"\n  no contact within {max_approach} s of approach, stopping")
                    break
                if self.contact_time is not None and self.data.time >= self.contact_time + after:
                    break
        progress.finish(self.data.time, i)
        self.wall_time = _time.perf_counter() - wall_start
        self.n = i
        self.log = {k: v[:i] for k, v in log.items()}
        self._frames = frames
        print(f"  {self.data.time:.3f} s simulated in {self.wall_time:.1f} s ({i} steps, "
              f"{self.wall_time / max(self.data.time, 1e-9):.1f} s per simulated second)")
        return self

    # ----------------------------------------------------------------------------------------- render
    def _renderer(self, height, width):
        """Offscreen renderer, camera and visual options shared by the video and the snapshots."""
        # the offscreen buffer must be at least as large as the image
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        renderer = mujoco.Renderer(self.model, height, width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.x_rest.mean(axis=0)
        camera.distance = 2.5 * float(np.ptp(self.x_rest, axis=0).max())
        camera.azimuth, camera.elevation = 45, -20
        vis = mujoco.MjvOption()
        vis.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        # wireframe (flex edges have zero width with radius 0, so the whole scene is drawn as
        # wireframe instead), contact points and contact force vectors
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = True
        vis.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        vis.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # MuJoCo draws a force F as an arrow of F * map.force / meanmass metres: scale it to
        # VIDEO_FORCE_SCALE m/N, and the markers to a fraction of the mean body size (~9 mm)
        self.model.vis.map.force = CONFIG.VIDEO_FORCE_SCALE * self.model.stat.meanmass
        self.model.vis.scale.contactwidth, self.model.vis.scale.contactheight = 0.1, 0.03
        self.model.vis.scale.forcewidth = 0.03
        return renderer, camera, vis

    def save_snapshot_images(self):
        """Render the stored states (first contact, max indentation) from an oblique and a side view
        (side by side in one png per moment). Returns the saved paths."""
        if not getattr(self, "_snap_states", None):
            print("  no snapshots (no contact)")
            return []
        from PIL import Image
        renderer, camera, vis = self._renderer(*CONFIG.SNAPSHOT_SIZE)
        data = mujoco.MjData(self.model)
        paths = []
        for name, state in self._snap_states.items():
            mujoco.mj_setState(self.model, data, state, self._snap_spec)
            mujoco.mj_forward(self.model, data)   # contacts and forces for the stored positions
            views = []
            for azimuth, elevation in CONFIG.SNAPSHOT_VIEWS:
                camera.azimuth, camera.elevation = azimuth, elevation
                renderer.update_scene(data, camera, vis)
                views.append(renderer.render().copy())
            depth = (self.contact_q - data.qpos[self.tool_qadr]) * 1e3
            path = self.outdir / f"{self.case}_snapshot_{name}.png"
            Image.fromarray(np.concatenate(views, axis=1)).save(path)
            print(f"saved snapshot {path} (t = {data.time:.3f} s, indentation {depth:.2f} mm)")
            paths.append(path)
        renderer.close()
        return paths

    # ------------------------------------------------------------------------------------------ save
    def results(self):
        """Recorded measures + time/depth logs (every SAVE_EVERY-th step) and metadata."""
        every = CONFIG.SAVE_EVERY
        s = slice(0, self.n, every)
        lg = self.log
        t = lg["time"]
        ct = self.contact_time
        t_rel = t - ct if ct is not None else t - t[-1] - 1.0
        depth = np.where(t >= ct, self.contact_q - lg["tool_q"], 0.0) if ct is not None else np.zeros_like(t)
        depth_target = (np.where(t >= ct, self.contact_target - lg["q_target"], 0.0)
                        if ct is not None else np.zeros_like(t))
        res = self.measures.results(self.n, every)
        res.update({"time": t[s], "t_rel": t_rel[s], "depth": depth[s], "depth_target": depth_target[s]})
        res.update({k: v[s] for k, v in lg.items() if k not in ("time",)})
        if self.surface_vertices is not None:
            res["top_vert"] = self.surface_vertices
        meta = {"case": self.case, "description": self.settings.get("description", ""), "xml": str(self.xml),
                "material": self.material, "elastic3d": self.elastic3d, **self.params,
                "damping_beta": self.params["damping"], **self.protocol, "protocol": self.protocol["type"],
                "contact_time": ct, "save_every": every, "sim_time": float(self.data.time),
                "wall_time": self.wall_time, "nsteps": self.n, "base_name": self.settings["base"]["name"],
                "comsol": self.settings.get("comsol"), **self.overrides}
        return res, meta

    def save(self):
        print(f"Saving results to {self.outdir} ...")
        self.outdir.mkdir(parents=True, exist_ok=True)
        res, meta = self.results()
        path = self.outdir / f"{self.case}_results.json"
        save_data(path, res, meta)
        # try: 
        #     (self.outdir / f"{self.case}_model.xml").write_text(self.spec.to_xml())   # the model as simulated
        if self._frames:
            from PIL import Image
            gif = self.outdir / f"{self.case}.gif"
            imgs = [Image.fromarray(f) for f in self._frames]
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / CONFIG.VIDEO_FPS), loop=0)
            print(f"saved video {gif} ({len(imgs)} frames)")
        if self.save_snapshots:
            self.save_snapshot_images()
        return path
