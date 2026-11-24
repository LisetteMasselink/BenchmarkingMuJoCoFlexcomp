"""Per-step recording of the validation measures for a rigid tool pressing on a pinned flex.

Used by the case setup scripts (tool = sphere / plates / exo, soft body = flex, base = what the
flex is pinned to). Call record(i) right after mj_step every step, then results(n) at the end.

Measures (world frame, SI units):
  contact_force   force of the tool on the soft body (N)
  P_from_tool     power out of the rigid tool: its contact force . its velocity (W)
  P_into_tissue   power into the soft body through the contact = -int(nI dA) in COMSOL:
                  the constraint forces (qfrc_constraint) on the flex vertices . vertex velocities (W);
                  equal to the tool contact power as long as nothing else constrains the flex (a
                  warning is printed otherwise)
  W_in, W_into    time integrals of the two (J); their difference E_contact is the energy taken
                  up by the contact itself (overlap + contact damping) = COMSOL contact energy
  E_overlap       sum over contacts of 1/2 normal force * overlap depth (stored part of E_contact)
  U_balance       elastic strain energy, -int(elastic force . velocity) dt (any material)
  D_material      energy dissipated by the material damping (J)
  KE              kinetic energy of the free flex vertices (J)
  W_gravity       work done by gravity (and other bias forces) on the free flex vertices (J)
  force_to_base   force the soft body passes to the base: through the pinned vertices (minus the
                  passive force on all free vertices; pinned vertices have no dofs, so MuJoCo does not
                  report their reaction) plus direct base-flex contact acting on free vertices
  P_surface       optional: P_into_tissue restricted to given surface vertices, with the flux
                  density -nI per vertex (W/m^2) stored at the moment of highest inward power
  U_strain        elastic strain energy from the current shape, computed as MuJoCo does from its
                  stiffness matrix model.flex_stiffness (see FlexElasticity). A state function, so unlike
                  U_balance it is not affected by contact chatter or the integrator's energy error
  base_load_pins  reaction force on every pinned vertex at the moment of peak tool force = how the load
                  is spread over the base/bone (element elastic forces summed onto the pinned vertices)
Energy balance of the soft body: W_into + W_gravity = U_balance + D_material + KE + residual,
where the residual is MuJoCo's numerical energy loss.
"""
import mujoco
import numpy as np


class FlexElasticity:
    """Strain energy and elastic vertex forces of a 3D flex (tetrahedra), exactly as MuJoCo
    computes them, from vertex positions (MuJoCo src/engine/engine_core_util.h).

    Kinematics (per tetrahedron, vertices x0..x3, linear shape functions so F is constant per element):
      Ds = [x1-x0, x2-x0, x3-x0]   edge matrix in the current shape (3x3, edges as columns)
      Dm = same in the rest shape;  V0 = |det Dm| / 6 the rest volume
      F  = Ds Dm^-1                deformation gradient: maps a rest edge onto the deformed edge
      J  = det F                   volume ratio V / V0
    Lame parameters: mu = E / (2(1+nu)) (shear modulus), lam = E nu / ((1+nu)(1-2nu)).

    Material models, W = strain energy density per rest volume (J/m^3), P = dW/dF the first
    Piola-Kirchhoff stress (force per rest area):
      elastic3d = 0, Saint Venant-Kirchhoff (StVK)
          E = (F'F - I) / 2                 Green-Lagrange strain (zero for any rigid rotation)
          W = mu E:E + lam_svk/2 (tr E)^2   lam_svk = lam from MuJoCo 4.0 on (textbook StVK),
                                            2 lam before (MuJoCo < 4.0 doubled the volume term)
          S = dW/dE = 2 mu E + lam_svk tr(E) I   second Piola-Kirchhoff stress
          P = F S                            (chain rule, dE/dF gives F)
          lam_svk is read from model.flex_stiffness, so it always matches the running MuJoCo.
          Linear elasticity extended to large rotations; it softens under strong compression and
          does not resist inversion (W does not grow as J -> 0).
      elastic3d = 1, stable neo-Hookean (SNH, Smith et al. 2018 type)
          Ic = tr(F'F) = F:F                 first invariant (sum of squared stretches)
          W = mu/2 (Ic - 3) - mu (J-1) + (lam+mu)/2 (J-1)^2
          P = mu F + (-mu + (lam+mu)(J-1)) cof(F)    using dIc/dF = 2F and dJ/dF = cof(F) = J F^-T
          At rest (F = I, J = 1) W = 0 and P = 0; the -mu (J-1) term cancels the shear term's
          stress there. No log(J), so it stays finite when an element is flattened or inverted.
      Both reduce to linear elasticity (sigma = 2 mu eps + lam tr(eps) I) at small strain (for StVK
      on MuJoCo < 4.0 with 2 lam instead of lam).

    Vertex forces: the element's energy is U = V0 W(F) and f_a = -dU/dx_a. Because F = Ds Dm^-1,
      dU/dDs = V0 P Dm^-T,  so  H = -V0 P Dm^-T
    holds the forces on vertices 1, 2, 3 in its columns. Vertex 0 gets minus their sum (the forces
    of an element sum to zero: a rigid translation does not change F). Forces of all elements that
    share a vertex are summed.

    Energy (energy, density): computed as MuJoCo does, from its stiffness matrix model.flex_stiffness
    (24 entries k per element; SNHElementEnergy in MuJoCo's tests). With s_i = l_i^2 - l0_i^2 the change
    in squared length of the element's 6 edges:
      U = sum_{i<=j} (i == j ? 1/4 : 1/2) k[idx] s_i s_j               k[0:21], upper triangle of 6x6
          + k[22] (J - 1)^2 + k[21] det(G)                             SNH only (k[21:24] = 0 for StVK)
    with G the 3x3 Gram matrix of the 3 edges at the element's first vertex (edges 0, 2, 4), built from
    the s values by the law of cosines. Equal to V0 W(F) above (relative difference ~1e-15, see
    test_measures.py).

    Forces (forces) from P above; checked against MuJoCo's qfrc_spring on deformed meshes (relative
    difference 1e-10 or less)."""

    def __init__(self, model, flex_id, x_rest, young, poisson, elastic3d):
        n = model.flex_elemnum[flex_id]
        adr = model.flex_elemdataadr[flex_id]
        self.elem = model.flex_elem[adr:adr + 4 * n].reshape(-1, 4)   # 4 vertex ids per tetrahedron
        self.nvert = model.flex_vertnum[flex_id]
        Dm = self._edges(x_rest)
        self.V0 = np.abs(np.linalg.det(Dm)) / 6                        # tetrahedron volume = 1/6 |det|
        self.Dm_inv = np.linalg.inv(Dm)
        self.Dm_invT = np.transpose(self.Dm_inv, (0, 2, 1))
        self.mu = young / (2 * (1 + poisson))
        self.lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.elastic3d = elastic3d

        # MuJoCo's stiffness matrix (n, 24), the 6 edge ids of every element, edge vertex pairs
        sadr = model.flex_stiffnessadr[flex_id]
        self.k = model.flex_stiffness[sadr:sadr + 24 * n].reshape(n, 24)   # stride 24 for 3D flexes
        eadr = model.flex_elemedgeadr[flex_id]
        self.elem_edges = model.flex_elemedge[eadr:eadr + 6 * n].reshape(n, 6)   # edge ids within the flex
        edgeadr, edgenum = model.flex_edgeadr[flex_id], model.flex_edgenum[flex_id]
        self.edge = model.flex_edge[edgeadr:edgeadr + edgenum]     # (vertex, vertex) per edge, local ids
        self.length0_sq = model.flexedge_length0[edgeadr:edgeadr + edgenum] ** 2
        self.triu = np.triu_indices(6)                             # order of k[0:21]: (0,0), (0,1), ..., (5,5)
        self.triu_weight = np.where(self.triu[0] == self.triu[1], 0.25, 0.5)

        # StVK volume coefficient used by this MuJoCo: from the energy at F = 2 I, where E = e I with e = 3/2,
        # W = 3 mu e^2 + lam_svk/2 (3 e)^2
        self.lam_svk = None
        if elastic3d == 0:
            e = 1.5
            W = np.mean(self.element_energy(2 * x_rest) / self.V0)
            self.lam_svk = 2 * (W - 3 * self.mu * e ** 2) / (3 * e) ** 2

    def _edges(self, x):
        """Edge matrices [x1-x0, x2-x0, x3-x0] of all elements, shape (n, 3, 3), edges as columns."""
        return np.stack([x[self.elem[:, k]] - x[self.elem[:, 0]] for k in (1, 2, 3)], axis=2)

    def _F(self, x):
        """Deformation gradient F = Ds Dm^-1 of all elements."""
        # reference material Dm (uneformed)
        # Deformed material  
        return self._edges(x) @ self.Dm_inv # deformation gradient I 

    @staticmethod
    def _cofactor(F):
        """cof(F) = J F^-T = dJ/dF, from cross products of the columns (finite even when J -> 0).
        Column k is perpendicular to the other two columns of F, scaled so that F' cof(F) = J I."""
        c0, c1, c2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        return np.stack([np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)], axis=2)

    def element_energy(self, x):
        """Strain energy of every element (J), from MuJoCo's stiffness matrix as MuJoCo computes it."""
        # based on https://github.com/google-deepmind/mujoco/blob/main/test/engine/engine_passive_test.cc
        l_sq = np.sum((x[self.edge[:, 0]] - x[self.edge[:, 1]]) ** 2, axis=1)
        s = (l_sq - self.length0_sq)[self.elem_edges]              # (n, 6) change in squared edge length
        i, j = self.triu
        energy = np.sum(self.triu_weight * self.k[:, :21] * s[:, i] * s[:, j], axis=1)
        if self.elastic3d == 0: # StVK: only the 21 quadratic entries
            return energy
        else: # energy SNH 
            J = np.linalg.det(self._F(x)) # SNH: volume terms with k[21], k[22]
            a, b, c = s[:, 0], s[:, 2], s[:, 4]
            d, e, f = 0.5 * (a + b - s[:, 1]), 0.5 * (a + c - s[:, 5]), 0.5 * (b + c - s[:, 3])
            return (energy + self.k[:, 22] * (J - 1) ** 2
                + self.k[:, 21] * (a * b * c + 2 * d * e * f - a * f * f - b * e * e - c * d * d))

    def density(self, x):
        """Strain energy density W per element (J/m^3, per rest volume) = element energy / V0."""
        return self.element_energy(x) / self.V0

    def energy(self, x):
        """Total strain energy (J): sum over the elements."""
        return self.element_energy(x).sum()

    def forces(self, x):
        """Elastic force -dU/dx on every flex vertex (N, world frame), pinned vertices included."""
        F = self._F(x)
        if self.elastic3d == 0:                                          # StVK
            E = 0.5 * (np.transpose(F, (0, 2, 1)) @ F - np.eye(3))
            S = 2 * self.mu * E + self.lam_svk * np.trace(E, axis1=1, axis2=2)[:, None, None] * np.eye(3)
            P = F @ S                                                    # first Piola-Kirchhoff stress
        else:                                                            # SNH
            J = np.linalg.det(F)
            cof = self._cofactor(F)
            P = self.mu * F + (-self.mu + (self.lam + self.mu) * (J - 1))[:, None, None] * cof
        H = -self.V0[:, None, None] * P @ self.Dm_invT                  # columns: forces on vertices 1, 2, 3
        out = np.zeros((self.nvert, 3))
        for k in (1, 2, 3):                                              # add.at: vertices shared by elements
            np.add.at(out, self.elem[:, k], H[:, :, k - 1])
        np.add.at(out, self.elem[:, 0], -H.sum(axis=2))                  # vertex 0: forces sum to zero
        return out


class ContactMeasures:
    """Records the measures of the module docstring every step; see record() for the order.

    model, data       MuJoCo model and data (call mj_forward first: the rest shape is taken from data)
    flex_id           the soft body
    tool_geom_ids     geoms of the rigid tool, tool_dof the dof of its (slide) joint
    nsteps            number of steps to reserve
    base_geom_ids     geoms of the base the flex is pinned to (for direct base-flex contact)
    surface_vertices  optional local flex vertex ids of the loaded surface, with their area (m^2) in
    surface_area      surface_area, for the flux density -nI per vertex
    material          (young, poisson, elastic3d) of the flex, for U_strain and the load on the pins
    """

    def __init__(self, model, data, flex_id, tool_geom_ids, tool_dof, nsteps,
                 base_geom_ids=(), surface_vertices=None, surface_area=None, material=None):
        self.model = model
        self.data = data 
        self.dt = model.opt.timestep
        self.flex_id = flex_id
        self.tool_dof = tool_dof
        self.tool_geoms = np.array(list(tool_geom_ids))
        self.base_geoms = np.array(list(base_geom_ids))

        # flex vertices: every free vertex is a body with 3 slide joints, a pinned vertex has no dofs
        adr = model.flex_vertadr[flex_id] # starting number of vertices id 
        num = model.flex_vertnum[flex_id] # number of vertices 
        self.vert_slice = slice(adr, adr + num)
        vert_body = model.flex_vertbodyid[adr:adr + num] # the vertices bodies

        vert_dofs = lambda b: np.arange(model.body_dofadr[b], model.body_dofadr[b] + model.body_dofnum[b]) # degrees of freedom for all vertices 
        self.dofs = np.concatenate([vert_dofs(b) for b in vert_body if model.body_dofnum[b] > 0]) # count all dofs 
        self.dof_mass = model.body_mass[model.dof_bodyid[self.dofs]] 
        self.dof_jnt = model.dof_jntid[self.dofs]
        self.pinned = np.where(model.body_dofnum[vert_body] == 0)[0] # find bodies which don't have degrees of freedom 
        self.x_rest = data.flexvert_xpos[self.vert_slice].copy()

        # optional parts
        self.surface = None
        if surface_vertices is not None:     # 3 dofs per surface vertex, for the flux per vertex
            self.surface = np.array([vert_dofs(vert_body[v]) for v in surface_vertices])
            self.surface_area = np.asarray(surface_area, dtype=float)
        self.elastic = FlexElasticity(model, flex_id, self.x_rest, *material) if material is not None else None

        # recorded values, one row per step
        vec, scalar = lambda: np.zeros((nsteps, 3)), lambda: np.zeros(nsteps)
        self.values = {
            # tool contact
            "contact_force": vec(), 
            "ncon_tool": scalar(), 
            "E_overlap": scalar(),
            # power and work through the tool contact
            "P_from_tool": scalar(), 
            "P_into_tissue": scalar(), 
            "P_surface": scalar(),
            "W_in": scalar(), 
            "W_into": scalar(),
            # energies of the soft body
            "U_strain": np.full(nsteps, np.nan), 
            "U_balance": scalar(), 
            "D_material": scalar(),
            "KE": scalar(), 
            "W_gravity": scalar(),
            # force to the base
            "force_base_pins": vec(), "force_base_contact": vec(), "force_to_base": vec(),
        }
        # snapshots: at peak inward surface power, at peak tool force and at the deepest tool position
        self.flux_peak = self.t_flux_peak = None
        self.base_load = self.t_base_load = None
        self.peak_tool_fz = 0.0
        self.deep_xpos = self.t_deep = None
        self.deep_tool_q = np.inf
        self._warned = False   # other constraints than the tool contacts found (see _check_constraints)

    # --------------------------------------------------------------------------------------- per step
    def record(self, i, tool_qpos):
        """Record step i; call right after mj_step. qvel is then the velocity over the step just taken,
        so force . qvel * dt is the work done in that step."""
        self._tool_contact(i) # record contact forces of tool on the flex
        self._power(i) # power of the constraint forces on the flex and the tool
        self._energies(i)
        self._base_force(i)
        self._check_constraints(i)
        self._snapshots(i, tool_qpos)

    def _tool_contact(self, i):
        """Total contact force of the tool on the flex, number of contacts, stored overlap energy."""
        force, n, overlap = self._contact_force_sum(self.tool_geoms)
        self.values["contact_force"][i] = force
        self.values["ncon_tool"][i] = n
        self.values["E_overlap"][i] = overlap

    def _power(self, i):
        """Power out of the tool and into the flex (= -int nI dA), their work, and the flux per surface vertex.
        qfrc_constraint = J^T efc_force of all constraints, on every dof: on the flex dofs that is the force of
        the tool contacts on the flex vertices, on the tool dof the (opposite) force on the tool, as long as
        no other constraint acts on them (checked in _check_constraints)."""
        self.values["P_into_tissue"][i] = self.data.qfrc_constraint[self.dofs] @ self.data.qvel[self.dofs] # flex side: forces on the vertices . vertex velocities
        self.values["P_from_tool"][i] = -self.data.qfrc_constraint[self.tool_dof] * self.data.qvel[self.tool_dof] # tool side (Fab = -Fba)

        self.values["W_in"][i] = self._integral("W_in", i, self.values["P_from_tool"][i])
        self.values["W_into"][i] = self._integral("W_into", i, self.values["P_into_tissue"][i])

        if self.surface is not None:
            P_vertex = np.sum(self.data.qfrc_constraint[self.surface] * self.data.qvel[self.surface], axis=1)

            self.values["P_surface"][i] = P_vertex.sum()
            if i == 0 or self.values["P_surface"][i] > self.values["P_surface"][:i].max():
                self.flux_peak, self.t_flux_peak = P_vertex / self.surface_area, self.data.time

    def _energies(self, i):
        """Strain energy from the shape (U_strain) and the energy balance terms of the soft body."""
        if self.elastic is not None:
            self.values["U_strain"][i] = self.elastic.energy(self.data.flexvert_xpos[self.vert_slice])
        self.values["U_balance"][i] = self._integral("U_balance", i, -self.data.qfrc_spring[self.dofs] @ self.data.qvel[self.dofs]) # spring energy
        self.values["D_material"][i] = self._integral("D_material", i, -self.data.qfrc_damper[self.dofs] @ self.data.qvel[self.dofs]) # damper energy
        self.values["W_gravity"][i] = self._integral("W_gravity", i, -self.data.qfrc_bias[self.dofs] @ self.data.qvel[self.dofs]) # gravity energy
        self.values["KE"][i] = 0.5 * np.sum(self.dof_mass * self.data.qvel[self.dofs] ** 2)

    def _base_force(self, i):
        """Force of the soft body on the base: through the pinned vertices plus direct base-flex contact.
        Pinned vertices have no dofs, so MuJoCo reports no reaction there: their load is minus the passive
        (elastic + damping) force on all free vertices."""
        
        self.values["force_base_pins"][i] = -self._to_world(self.data.qfrc_passive)
        if len(self.base_geoms):
            self.values["force_base_contact"][i] = -self._contact_force_sum(self.base_geoms)[0]
        self.values["force_to_base"][i] = self.values["force_base_pins"][i] + self.values["force_base_contact"][i]

    def _check_constraints(self, i):
        """qfrc_constraint holds all constraints. Warn (once) when something else than the tool contacts acts on
        the flex or the tool (base contacts, a joint limit, ...): P_into_tissue / P_from_tool then include it."""
        if self._warned:
            return
        tool_contact = self.values["contact_force"][i]             # on the flex
        expected_flex = tool_contact - self.values["force_base_contact"][i]
        expected_tool = -tool_contact @ self.data.xaxis[self.model.dof_jntid[self.tool_dof]]
        tol = 1e-6 + 1e-4 * np.linalg.norm(tool_contact)
        if (np.linalg.norm(self._to_world(self.data.qfrc_constraint) - expected_flex) > tol
                or abs(self.data.qfrc_constraint[self.tool_dof] - expected_tool) > tol
                or np.any(self.values["force_base_contact"][i])):
            print(f"\n  WARNING t = {self.data.time:.4f} s: other constraints than the tool contacts act on the "
                  f"flex or tool (base contact / joint limit): P_into_tissue and P_from_tool include them")
            self._warned = True

    def _snapshots(self, i, tool_qpos):
        """Load on every pinned vertex at peak tool force, shape of the flex at the deepest tool position."""
        fz = abs(self.values["contact_force"][i][2])
        if self.elastic is not None and fz > self.peak_tool_fz:
            self.peak_tool_fz = fz 
            self.t_base_load = self.data.time
            self.base_load = self.elastic.forces(self.data.flexvert_xpos[self.vert_slice])[self.pinned]
        if tool_qpos < self.deep_tool_q:
            self.deep_tool_q = tool_qpos 
            self.t_deep = self.data.time 
            self.deep_xpos = self.data.flexvert_xpos[self.vert_slice].copy()

    # ---------------------------------------------------------------------------------------- helpers
    def _integral(self, key, i, power):
        """Running time integral: value of the previous step + power * dt. (resulting in energy)""" 
        return (self.values[key][i - 1] if i > 0 else 0.0) + power * self.dt

    def _contacts(self, geoms):
        """Indices of the active contacts between the flex and any of the given geoms."""
        hit = (self.data.contact.flex == self.flex_id).any(axis=1) & np.isin(self.data.contact.geom, geoms).any(axis=1)
        return np.flatnonzero(hit)

    def _contact_force_sum(self, geoms):
        """Summed force (world) of the flex contacts with the given geoms, acting on the flex; their number;
        and the stored overlap energy (1/2 normal force * overlap depth)."""
        force, overlap, ft = np.zeros(3), 0.0, np.zeros(6)
        contacts = self._contacts(geoms)
        for j in contacts:
            c = self.data.contact[j]
            mujoco.mj_contactForce(self.model, self.data, j, ft)   # in the contact frame, normal first
            on_flex = 1.0 if c.flex[1] == self.flex_id else -1.0     # the force acts on the 2nd of the pair
            force += on_flex * c.frame.reshape(3, 3).T @ ft[:3]
            overlap += 0.5 * ft[0] * max(-c.dist, 0.0)
        return force, len(contacts), overlap

    def _to_world(self, qfrc):
        """Net force (world) of a generalized force on the free flex vertices (their slide axes)."""
        return (self.data.xaxis[self.dof_jnt] * qfrc[self.dofs, None]).sum(axis=0)

    # ----------------------------------------------------------------------------------------- output
    def results(self, n, every=1):
        """Recorded values truncated to n steps and decimated, plus the snapshots."""
        out = {k: v[0:n:every] for k, v in self.values.items()}
        out.update({"X_deep": self.deep_xpos, "t_deep": self.t_deep,
                    "X_rest_flex": self.x_rest, "pinned": self.pinned,
                    "base_load_pins": self.base_load, "t_base_load": self.t_base_load})
        if self.surface is not None:
            out.update({"flux_peak": self.flux_peak, "t_flux_peak": self.t_flux_peak,
                        "surface_area": self.surface_area})
        return out
