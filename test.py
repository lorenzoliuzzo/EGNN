import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import MessagePassing


def create_lattice(shape):
    dims = len(shape)
    num_nodes = np.prod(shape)
    grid = np.arange(num_nodes).reshape(shape)
    
    srcs, dsts, dirs, fwd = [], [], [], []

    # 1. Build the raw lists
    for d in range(dims):
        for is_fwd in [True, False]:
            shift = -1 if is_fwd else 1
            neighbor_grid = np.roll(grid, shift, axis=d)
            
            s_list = grid.flatten()
            d_list = neighbor_grid.flatten()
            
            for i in range(num_nodes):
                srcs.append(s_list[i])
                dsts.append(d_list[i])
                dirs.append(d)
                fwd.append(is_fwd)

    # 2. Convert to tensors
    srcs_t = torch.tensor(srcs, dtype=torch.long)
    dsts_t = torch.tensor(dsts, dtype=torch.long)
    dirs_t = torch.tensor(dirs, dtype=torch.long)
    fwd_t = torch.tensor(fwd, dtype=torch.bool)
    
    # 3. THE OPTIMIZATION: Sort edges by Direction, then Forward/Backward
    # This guarantees contiguous memory chunks for the GPU
    sort_keys = dirs_t * 2 + (~fwd_t).long()
    sorted_idx = torch.argsort(sort_keys)
    
    srcs_sorted = srcs_t[sorted_idx]
    dsts_sorted = dsts_t[sorted_idx]
    edge_dirs = dirs_t[sorted_idx]
    is_fwd = fwd_t[sorted_idx]
    
    edge_index = torch.stack([srcs_sorted, dsts_sorted], dim=0)
    
    # 4. Rebuild the Partner Map using the sorted edges
    # We use a dictionary to find the reverse edge (dst -> src)
    edge_dict = {(s.item(), d.item()): i for i, (s, d) in enumerate(zip(srcs_sorted, dsts_sorted))}
    partner_map = torch.tensor([edge_dict[(d.item(), s.item())] for s, d in zip(srcs_sorted, dsts_sorted)], dtype=torch.long)
    
    return edge_index, edge_dirs, is_fwd, partner_map


def find_rectangular_loops(lattice_shape, edge_index, R=1, T=1):
    """
    Vectorized finder for R x T rectangular loops in an N-D periodic lattice.
    Returns: A tensor of shape [Num_Loops, 2*R + 2*T] containing the ordered edge indices.
    """
    dims = len(lattice_shape)
    V = np.prod(lattice_shape)
    device = edge_index.device # Detect where the lattice lives
    all_loops = []
    
    # Base starting nodes [0, 1, ..., V-1] on the correct device
    base_nodes = torch.arange(V, dtype=torch.long, device=device)
    
    for mu in range(dims):
        for nu in range(mu + 1, dims):
            orientations = [(mu, nu, R, T)]
            if R != T:
                orientations.append((mu, nu, T, R))
                
            for dir1, dir2, len1, len2 in orientations:
                current_nodes = base_nodes.clone()
                loop_edges = []
                
                # Walk the rectangle
                for d, length in [(dir1, len1), (dir2, len2)]:
                    for _ in range(length):
                        e = (d * 2 * V) + current_nodes
                        loop_edges.append(e)
                        current_nodes = edge_index[1, e]
                        
                # Walk back (-dir)
                for d, length in [(dir1, len1), (dir2, len2)]:
                    for _ in range(length):
                        e = (d * 2 * V) + V + current_nodes
                        loop_edges.append(e)
                        current_nodes = edge_index[1, e]
                    
                plane_loops = torch.stack(loop_edges, dim=1)
                all_loops.append(plane_loops)

    return torch.cat(all_loops, dim=0)


def get_gate(raw_u, partner_map, is_fwd, is_su=True):
    """
    Projects raw tensors into Unitary (U(N)) or Special Unitary (SU(N)) groups.
    """
    # 1. Ensure the generator is anti-Hermitian (H = -H^dagger)
    h = (raw_u - raw_u.adjoint()).resolve_conj()
    
    # 2. If SU(N), subtract the trace to ensure det(exp(H)) = 1
    if is_su:
        dim = h.shape[-1]
        # Calculate trace for each edge matrix
        trace = torch.einsum('eii->e', h)
        # Identity matrix for the specific dimension
        eye = torch.eye(dim, dtype=torch.complex64, device=h.device)
        # Reshape trace to [Edges, 1, 1] for broadcasting
        h = h - (trace.view(-1, 1, 1) / dim) * eye
    
    # 3. Exponential map: H (Lie Algebra) -> U (Lie Group)
    u_all = torch.matrix_exp(h)
    
    # 4. Enforce U(-mu) = U(mu)^dagger for lattice consistency
    u_bwd_calculated = u_all[partner_map].adjoint().resolve_conj()
    u_consistent = torch.where(is_fwd.view(-1, 1, 1), u_all, u_bwd_calculated)
    
    return u_consistent


class GaugeEquivariantConv(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='add', node_dim=0)
        self.lin = nn.Linear(channels, channels, bias=False, dtype=torch.complex64)

    def forward(self, x, edge_index, u_gate):
        # x: [Nodes, Channels, Group_Dim]
        # u_gate: [Edges, Group_Dim, Group_Dim]
        out = self.propagate(edge_index, x=x, u_gate=u_gate)
        
        # Mix channels: transpose to [Nodes, Group_Dim, Channels] 
        # so Linear only touches the Channel dimension.
        out = out.transpose(1, 2)
        out = self.lin(out)
        return out.transpose(1, 2)

    def message(self, x_j, u_gate):
        # Parallel Transport: U_ij * psi_j
        # u_gate: [E, G, G], x_j: [E, C, G]
        return torch.einsum('egh, ech -> ecg', u_gate, x_j)


class WilsonAction(nn.Module):
    def __init__(self, plaq_idx, group_dim, beta):
        super().__init__()
        self.register_buffer('plaq_idx', plaq_idx)
        self.group_dim = group_dim
        self.beta = beta

    def forward(self, u_gates):
        """
        u_gates: [Total_Edges, dim, dim]
        """
        # Get all edges for all plaquettes at once
        # Shape: [Num_Plaquettes, 4, dim, dim]
        u_p = u_gates[self.plaq_idx]
        
        # Matrix multiply around the loop: U1 * U2 * U3 * U4
        # We can use a loop over the 4 edges of the plaquette
        loop = u_p[:, 0]
        for i in range(1, 4):
            loop = torch.matmul(loop, u_p[:, i])
        tr = torch.einsum('pii -> p', loop)
        
        # 4. Final Scalar calculation
        # S = beta * sum(1 - Re(Tr(U))/Nc)
        nc = float(self.group_dim)
        action_per_plaq = 1.0 - (tr.real / nc)
        
        return self.beta * torch.sum(action_per_plaq)


class HiggsAction(nn.Module):
    def __init__(self, v=1.0, lam=0.1):
        super().__init__()
        self.v = v
        self.lam = lam

    def forward(self, phi, edge_index, is_fwd, u_su2, u_u1):
        # 1. Potential
        phi_sq = torch.sum(phi.abs()**2, dim=-1)
        v_pot = self.lam * torch.sum((phi_sq - self.v**2)**2)
        
        # 2. Kinetic Interaction with BOTH fields
        src, dst = edge_index[0, is_fwd], edge_index[1, is_fwd]
        
        u_total = u_su2[is_fwd] * u_u1[is_fwd]
        phi_j_transported = torch.einsum('eij, ej -> ei', u_total, phi[src])
        diff = phi[dst] - phi_j_transported
        kinetic = torch.sum(diff.abs()**2)

        # kinetic = torch.sum(torch.norm(phi[dst] - phi_j_transported, dim=-1)**2)        

        # src, dst = edge_index[0], edge_index[1]
        # phi_j = phi[src]
        # phi_i = phi[dst]
        
        # # Combine the forces: U_total = U_su2 * U_u1
        # # Since U_u1 is a scalar phase (1x1), we can just multiply
        # u_total = u_su2 * u_u1
        
        # phi_j_transported = torch.einsum('egk, ek -> eg', u_total, phi_j)
        # kinetic = torch.sum(torch.norm(phi_i - phi_j_transported, dim=-1)**2)
        
        return kinetic + v_pot


class WilsonDiracOperator(nn.Module):
    def __init__(self, y=1.0, bare_mass=0.01, r=1.0):
        super().__init__()
        self.y = y                 # Yukawa coupling constant
        self.bare_mass = bare_mass # Tiny stabilizer for the CG solver
        self.r = r
        
        # 1. Precompute Gamma matrices
        g1, g2, g3, g4 = self._gamma_matrices()
        gammas = torch.stack([g1, g2, g3, g4], dim=0)
        self.register_buffer('gammas', gammas)
        
        g5 = g1 @ g2 @ g3 @ g4
        self.register_buffer('g5', g5)
        self.register_buffer('id_spin', torch.eye(4, dtype=torch.complex64))


    def _gamma_matrices(self):
        # Standard Euclidean Gamma matrices (Chiral representation)
        zeros = torch.zeros(2, 2, dtype=torch.complex64)
        eye = torch.eye(2, dtype=torch.complex64)
        px = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        py = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        pz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        # Gamma 1, 2, 3 (Spatial)
        g1 = torch.cat([torch.cat([zeros, 1j*px], dim=1), torch.cat([-1j*px, zeros], dim=1)], dim=0)
        g2 = torch.cat([torch.cat([zeros, 1j*py], dim=1), torch.cat([-1j*py, zeros], dim=1)], dim=0)
        g3 = torch.cat([torch.cat([zeros, 1j*pz], dim=1), torch.cat([-1j*pz, zeros], dim=1)], dim=0)
        # Gamma 4 (Temporal)
        g4 = torch.cat([torch.cat([zeros, eye], dim=1), torch.cat([eye, zeros], dim=1)], dim=0)
        
        return [g1, g2, g3, g4]


    # ADDED phi to the forward pass arguments
    def forward(self, psi, phi, edge_index, edge_dirs, u_su3, is_fwd):
        dim_space = len(torch.unique(edge_dirs))
        
        # --- NEW: DYNAMICAL YUKAWA MASS ---
        # phi is [Nodes, 2] (Complex SU(2) doublet)
        # Calculate the gauge-invariant local magnitude |phi| at every node
        phi_sq = torch.sum(phi.abs()**2, dim=-1)
        higgs_vev_local = torch.sqrt(phi_sq).view(-1, 1, 1) # Shape: [Nodes, 1, 1]
        
        # The mass is now a dynamic field dependent on the Higgs!
        dynamic_mass = self.bare_mass + (self.y * higgs_vev_local)
        out = (dynamic_mass + self.r * dim_space) * psi
        # ----------------------------------
        
        src, dst = edge_index[0], edge_index[1]
        
        # Parallel Transport (Color Rotation)
        rotated_psi = torch.einsum('ecd, eds -> ecs', u_su3, psi[src])
        
        # Vectorized Spin Gather
        g_edge = self.gammas[edge_dirs]
        
        # Spin Projection
        sign = torch.where(is_fwd, -1.0, 1.0).view(-1, 1, 1).to(psi.device)
        P_edge = self.r * self.id_spin + sign * g_edge
        
        # Multiply projection against the rotated spinor
        term = torch.einsum('eab, ecb -> eca', P_edge, rotated_psi)
        
        # Scatter Add
        out.index_add_(0, dst, -0.5 * term)
        
        return out


@torch.no_grad() 
def conjugate_gradient(dirac_op, psi, phi, edge_index, edge_dirs, u_su3, is_fwd, max_iter=200, tol=1e-6):
    x = torch.zeros_like(psi)
    
    def apply_m_dag_m(vec):
        # Pass phi into the Dirac operator
        d_vec = dirac_op(vec, phi, edge_index, edge_dirs, u_su3, is_fwd)
        g5_d_vec = torch.einsum('ss, ecs -> ecs', dirac_op.g5, d_vec)
        d_dag_partial = dirac_op(g5_d_vec, phi, edge_index, edge_dirs, u_su3, is_fwd)
        return torch.einsum('ss, ecs -> ecs', dirac_op.g5, d_dag_partial)

    r = psi - apply_m_dag_m(x)
    p = r.clone()
    rsold = torch.sum(r.conj() * r).real

    for i in range(max_iter):
        ap = apply_m_dag_m(p)
        alpha = rsold / (torch.sum(p.conj() * ap).real + 1e-10)
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = torch.sum(r.conj() * r).real
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    return x


class PseudofermionAction(nn.Module):
    def __init__(self, y=1.0):
        super().__init__()
        # Initialize with Yukawa coupling instead of fixed mass
        self.dirac = WilsonDiracOperator(y=y)

    # Added higgs_phi argument
    def forward(self, pf_phi, higgs_phi, edge_index, edge_dirs, u_su3, is_fwd):
        # 1. Solve x = (D^dag D)^-1 pf_phi (Detached for memory safety)
        x = conjugate_gradient(self.dirac, pf_phi, higgs_phi, edge_index, edge_dirs, u_su3, is_fwd)
        
        # 2. Calculate the actual physical action for logging
        actual_action = torch.sum(pf_phi.conj() * x).real
        
        # 3. The "Surrogate Gradient" Trick
        dx = self.dirac(x, higgs_phi, edge_index, edge_dirs, u_su3, is_fwd)
        surrogate_loss = -torch.sum(dx.abs()**2)
                
        return surrogate_loss - surrogate_loss.detach() + actual_action


class VacuumFinder(nn.Module):
    def __init__(self, lattice_shape, groups={'su3': 3, 'su2': 2, 'u1': 1}, 
                 betas={'su3': 6.0, 'su2': 4.0, 'u1': 5.0}, v=1.0, lam=0.5):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lattice_shape = lattice_shape
        self.groups = groups
        self.betas = betas
        
        # 1. Lattice Geometry Infrastructure
        edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_dirs', edge_dirs)
        self.register_buffer('partner_map', partner_map)
        self.register_buffer('is_fwd', is_fwd)
        
        plaq_idx = find_rectangular_loops(lattice_shape, edge_index) #, edge_dirs, is_fwd)
        self.register_buffer('plaq_idx', plaq_idx)

        # 2. Dynamic Gauge Field Parameters (The Standard Model Force Carriers)
        num_edges = edge_index.size(1)
        # self.u_raw = nn.ParameterDict({
        #     name: nn.Parameter(torch.randn(num_edges, dim, dim, dtype=torch.complex64))
        #     for name, dim in groups.items()
        # })

        # cold start
        self.u_raw = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(num_edges, dim, dim, dtype=torch.complex64))
            for name, dim in groups.items()
        })

        # 3. Matter Field (The Higgs Doublet)
        num_nodes = np.prod(lattice_shape)
        self.phi = nn.Parameter(torch.randn(num_nodes, 2, dtype=torch.complex64))
        
        # self.pf_phi = nn.Parameter(torch.randn(num_nodes, 3, 4, dtype=torch.complex64))
        self.register_buffer('pf_phi', torch.randn(num_nodes, 3, 4, dtype=torch.complex64))

        # 4. Action Calculators
        self.wilson_calcs = nn.ModuleDict({
            name: WilsonAction(self.plaq_idx, group_dim=dim, beta=betas[name])
            for name, dim in groups.items()
        })
        self.higgs_calc = HiggsAction(v=v, lam=lam)
        self.fermion_calc = PseudofermionAction(y=1.0)
        self.to(self.device)


    def find_vacuum(self, steps=1000):
        # 1. Multi-Optimizer Setup: Isolate parameters based on their physical energy scales
        # SU(3) is highly energetic; it gets a very slow, careful learning rate.
        opt_su3 = optim.Adam([self.u_raw['su3']], lr=0.001) 
        
        # Electroweak forces are relatively stable; they learn faster.
        opt_ew = optim.Adam([self.u_raw['su2'], self.u_raw['u1']], lr=0.005)
        
        # The Higgs VEV settles very smoothly; it gets the fastest learning rate.
        opt_higgs = optim.Adam([self.phi], lr=0.01)

        # 2. Granular History Tracking
        history = {
            'total': [], 'vev': [], 
            'gauge_su3': [], 'gauge_su2': [], 'gauge_u1': [], 
            'higgs': [], 'fermion': []
        }

        print(f"--- Cooling Vacuum (Multi-Optimizer Sector Cooling) ---")
        
        for step in range(steps):
            # A. Zero all gradients
            opt_su3.zero_grad()
            opt_ew.zero_grad()
            opt_higgs.zero_grad()
            
            # B. Enforce Manifold Constraints (Project to Unitary Group)
            physical_gates = {}
            for name, raw_u in self.u_raw.items():
                is_special = (name != 'u1')
                physical_gates[name] = get_gate(raw_u, self.partner_map, self.is_fwd, is_su=is_special)

            # C. Calculate Individual Action Components
            # 1. Gauge Actions
            s_gauge_su3 = self.wilson_calcs['su3'](physical_gates['su3'])
            s_gauge_su2 = self.wilson_calcs['su2'](physical_gates['su2'])
            s_gauge_u1  = self.wilson_calcs['u1'](physical_gates['u1'])
            s_gauge_total = s_gauge_su3 + s_gauge_su2 + s_gauge_u1

            # 2. Higgs Action (Coupled to SU2 and U1)
            s_higgs = self.higgs_calc(self.phi, self.edge_index, self.is_fwd,
                                      physical_gates['su2'], physical_gates['u1'])

            # 3. Fermion Action (Coupled to SU3)
            s_fermion = self.fermion_calc(self.pf_phi, self.phi, self.edge_index, 
                                          self.edge_dirs, physical_gates['su3'], self.is_fwd)

            # D. Total Energy & Backpropagation
            loss = s_gauge_total + s_higgs + s_fermion
            loss.backward()
            
            # E. Safe Complex Gradient Resolution (CRITICAL PyTorch Fix)
            # PyTorch accumulates complex gradients as conjugated views. 
            # We must resolve them in memory before Adam attempts to update the momentum.
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None and p.grad.is_complex():
                        p.grad = p.grad.resolve_conj()

            # F. Step Optimizers Independently
            opt_su3.step()
            opt_ew.step()
            opt_higgs.step()

            # G. The "Heat Bath" (Quark Resampling)
            # This prevents the optimizer from "deleting" the fermions.
            if step % 10 == 0:
                with torch.no_grad():
                    # phi = M^dag eta (M^dag = g5 M g5, eta ~ complex N(0,1)) so the
                    # pseudofermion action carries the fermion determinant weight
                    dirac = self.fermion_calc.dirac
                    g5_eta = torch.einsum('ss, ecs -> ecs', dirac.g5,
                                          torch.randn_like(self.pf_phi))
                    d_eta = dirac(g5_eta, self.phi, self.edge_index, self.edge_dirs,
                                  physical_gates['su3'], self.is_fwd)
                    self.pf_phi.copy_(torch.einsum('ss, ecs -> ecs', dirac.g5, d_eta))

            # H. Logging
            current_vev = torch.norm(self.phi, dim=-1).mean().item()
            history['total'].append(loss.item())
            history['vev'].append(current_vev)
            history['gauge_su3'].append(s_gauge_su3.item())
            history['gauge_su2'].append(s_gauge_su2.item())
            history['gauge_u1'].append(s_gauge_u1.item())
            history['higgs'].append(s_higgs.item())
            history['fermion'].append(s_fermion.item())
            
            # Print update to console
            if step % 100 == 0:
                print(f"Step {step:03d} | Total: {loss.item():.2f} | SU3: {s_gauge_su3.item():.2f} | Fermion: {s_fermion.item():.2f} | VEV: {current_vev:.3f}")

        return history
        
          
    def find_vacuum3(self, steps=1000, dt=0.005, thermalize_steps=200):
        """
        Simulates Quantum Fluctuations using Langevin Dynamics.
        dt: The integration time step.
        thermalize_steps: Steps discarded while the system reaches equilibrium.
        """
        history = {'total': [], 'vev': [], 'gauge': [], 'fermion': []}
        for name in self.groups.keys(): history[name] = []

        print(f"--- Quantum Thermalization (Langevin Dynamics) ---")
        
        for step in range(steps):
            # Manually zero gradients (no optimizer)
            self.zero_grad()
            
            # A. Enforce Manifold Constraints (Unitarity/Adjointness)
            physical_gates = {}
            for name, raw_u in self.u_raw.items():
                physical_gates[name] = get_gate(raw_u, self.partner_map, self.is_fwd)

            # B. Calculate Total Action
            s_gauge = 0
            for name, gate in physical_gates.items():
                s_group = self.wilson_calcs[name](gate)
                s_gauge += s_group

            s_higgs = self.higgs_calc(self.phi, self.edge_index, self.is_fwd,
                                      physical_gates['su2'], physical_gates['u1'])

            s_fermion = self.fermion_calc(self.pf_phi, self.edge_index, 
                                          self.edge_dirs, physical_gates['su3'], self.is_fwd)

            loss = s_gauge + s_higgs + s_fermion
            
            # C. Calculate Physical Forces (Backpropagation)
            loss.backward()

            # D. The Langevin Integration Step (Euler-Maruyama)
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None:
                        # 1. Extract the physical force
                        force = p.grad.resolve_conj()
                        
                        # 2. Generate Fluctuation-Dissipation Noise
                        if p.is_complex():
                            # For complex numbers, variance splits between real and imag.
                            # Standard derivation requires scaling by sqrt(dt) for complex
                            noise_r = torch.randn_like(p.real)
                            noise_i = torch.randn_like(p.imag)
                            noise = torch.complex(noise_r, noise_i) * np.sqrt(dt)
                        else:
                            # Real parameters scale by sqrt(2 * dt)
                            noise = torch.randn_like(p) * np.sqrt(2 * dt)
                        
                        # 3. Update the Field: U = U - Force*dt + Noise
                        p.sub_(force * dt)
                        p.add_(noise)

            # E. Heat Bath Resampling (Refresh the Pseudofermions)
            if step % 10 == 0:
                with torch.no_grad():
                    # Properly resample complex standard normal
                    self.pf_phi.real.normal_()
                    self.pf_phi.imag.normal_()
                    self.pf_phi.mul_(1.0 / np.sqrt(2))
                    
            # F. Logging & Thermalization Logic
            current_vev = torch.norm(self.phi, dim=-1).mean().item()
            
            # We ONLY record history AFTER the system has reached thermal equilibrium
            if step > thermalize_steps:
                history['total'].append(loss.item())
                history['vev'].append(current_vev)
                history['gauge'].append(s_gauge.item())
                history['fermion'].append(s_fermion.item())

            if step % 100 == 0:
                phase = "Thermalizing" if step <= thermalize_steps else "Sampling"
                print(f"[{phase}] Step {step:03d} | Action: {loss.item():.2f} | VEV: {current_vev:.3f}")

        return history

        
    def find_vacuum2(self, steps=1000, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        history = {'total': [], 'vev': [], 'gauge': [], 'fermion': []}
        for name in self.groups.keys(): history[name] = []

        print(f"--- Cooling Vacuum ({', '.join(self.groups.keys())}) ---")
        for step in range(steps):
            optimizer.zero_grad()
            
            # A. Enforce Manifold Constraints (Unitarity/Adjointness)
            physical_gates = {}
            for name, raw_u in self.u_raw.items():
                # factory = UniversalGateFactory()
                physical_gates[name] = get_gate(raw_u, self.partner_map, self.is_fwd)

            # B. Calculate Total Action
            # 1. Sum Wilson Actions for all gauge groups (Gluons + Electroweak)
            s_gauge = 0
            for name, gate in physical_gates.items():
                s_group = self.wilson_calcs[name](gate)
                s_gauge += s_group
                history[name].append(s_group.item())

            # 2. Calculate Higgs Interaction (Couples only to SU2 and U1)
            s_higgs = self.higgs_calc(self.phi, self.edge_index, self.is_fwd,
                                      physical_gates['su2'], 
                                      physical_gates['u1'])

            s_fermion = self.fermion_calc(self.pf_phi, self.edge_index, 
                                          self.edge_dirs, physical_gates['su3'], self.is_fwd)

            loss = s_gauge + s_higgs + s_fermion
            
            # C. Optimize
            loss.backward()
            optimizer.step()

            # D. Logging
            current_vev = torch.norm(self.phi, dim=-1).mean().item()
            history['total'].append(loss.item())
            history['vev'].append(current_vev)
            history['gauge'].append(s_gauge.item())
            history['fermion'].append(s_fermion.item())

            if step % 10 == 0:
                with torch.no_grad():
                    self.pf_phi.normal_(std=1.0) # Keep the quarks 'alive'
                    
            if step % 100 == 0:
                print(f"Step {step:03d} | Total Action: {loss.item():.4f} | Gauge: {s_gauge.item():.2f} | Fermion: {s_fermion.item():.2f} | VEV: {current_vev:.3f}")

        return history


class QuantumVacuumFinder(VacuumFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                            
        self.dt_map = {
            'su3': 0.0001,  # Stiffest/Hardest: Tiny steps to preserve SU(3) manifold
            'su2': 0.001,   # Moderate: Standard electroweak scale
            'u1':  0.005,   # Fast: Abelian field, easier to sample
            'phi': 0.01     # Higgs: Large steps to explore the VEV landscape
        }


    def langevin_step(self, parameters, dt):
        """
        Performs a Stochastic Gradient Descent step with the correct 
        noise scaling for quantum sampling.
        """
        with torch.no_grad():
            for p in parameters:
                if p.grad is None:
                    continue
                
                # 1. Resolve complex gradients (standard PyTorch requirement)
                grad = p.grad.resolve_conj()
                
                # 2. Generate the Quantum Noise (Diffusion)
                # Noise must be complex if the parameter is complex
                noise = torch.randn_like(p)
                if p.is_complex():
                    noise_imag = torch.randn_like(p)
                    noise = (noise + 1j * noise_imag) / np.sqrt(2)
                
                # 3. Apply Langevin Update: phi = phi - dt * grad + sqrt(2*dt) * noise
                drift = -dt * grad
                diffusion = np.sqrt(2 * dt) * noise
                p.add_(drift + diffusion)


    def find_vacuum(self, steps=1000):
        history = {
            'total': [], 'vev': [], 
            'gauge_su3': [], 'gauge_su2': [], 'gauge_u1': [], 
            'higgs': [], 'fermion': []
        }
        
        for step in range(steps):
            # A. Calculate Action (Forward Pass)
            physical_gates = {
                name: get_gate(raw, self.partner_map, self.is_fwd, is_su=(name != 'u1'))
                for name, raw in self.u_raw.items()
            }
            
            s_gauge_su3 = self.wilson_calcs['su3'](physical_gates['su3'])
            s_gauge_su2 = self.wilson_calcs['su2'](physical_gates['su2'])
            s_gauge_u1  = self.wilson_calcs['u1'](physical_gates['u1'])
            s_gauge = s_gauge_su3 + s_gauge_su2 + s_gauge_u1

            s_higgs = self.higgs_calc(self.phi, self.edge_index, self.is_fwd, 
                                      physical_gates['su2'], physical_gates['u1'])
            s_fermion = self.fermion_calc(self.pf_phi, self.phi, self.edge_index, 
                                          self.edge_dirs, physical_gates['su3'], self.is_fwd)
            
            total_action = s_gauge + s_higgs + s_fermion
            
            # B. Backward Pass to get Drifts (Gradients)
            self.zero_grad()
            total_action.backward()

            # ... (inside your training loop after total_action.backward()) ...

            # C. Multi-scale Stochastic Step
            # Update Gauge Fields individually based on their group type
            for name, param in self.u_raw.items():
                if name in self.dt_map:
                    self.langevin_step([param], dt=self.dt_map[name])

            # Update the Higgs field with its own specific time-step
            self.langevin_step([self.phi], dt=self.dt_map['phi'])

            # D. The Fermion Heat Bath
            if step % 10 == 0:
                with torch.no_grad():
                    # phi = M^dag eta (M^dag = g5 M g5, eta ~ complex N(0,1)) so the
                    # pseudofermion action carries the fermion determinant weight
                    dirac = self.fermion_calc.dirac
                    g5_eta = torch.einsum('ss, ecs -> ecs', dirac.g5,
                                          torch.randn_like(self.pf_phi))
                    d_eta = dirac(g5_eta, self.phi, self.edge_index, self.edge_dirs,
                                  physical_gates['su3'], self.is_fwd)
                    self.pf_phi.copy_(torch.einsum('ss, ecs -> ecs', dirac.g5, d_eta))


            # H. Logging
            current_vev = torch.norm(self.phi, dim=-1).mean().item()
            history['total'].append(total_action.item())
            history['vev'].append(current_vev)
            history['gauge_su3'].append(s_gauge_su3.item())
            history['gauge_su2'].append(s_gauge_su2.item())
            history['gauge_u1'].append(s_gauge_u1.item())
            history['higgs'].append(s_higgs.item())
            history['fermion'].append(s_fermion.item())
            
            # Print update to console
            if step % 100 == 0:
                print(f"Step {step:03d} | Total: {total_action.item():.2f} | Gauge: {s_gauge.item():.2f} | Fermion: {s_fermion.item():.2f} | VEV: {current_vev:.3f}")

        return history
        


def plot_action_landscape(history):
    """
    Plots the separate energy contributions to the vacuum.
    """
    epochs = range(len(history['total']))
    
    plt.figure(figsize=(10, 6))
    
    # 1. Total Action (The overarching landscape)
    plt.plot(epochs, history['total'], 'k-', linewidth=2, label='Total Action ($S_{tot}$)')
    
    # 2. Individual Sector Contributions
    # Note: You may need to scale these if one dominates the others by orders of magnitude
    plt.plot(epochs, history['gauge_su3'], 'r--', alpha=0.8, label='SU(3) Strong Action')
    plt.plot(epochs, history['gauge_su2'], 'g--', alpha=0.8, label='SU(2) Weak Action')
    plt.plot(epochs, history['gauge_u1'], 'b--', alpha=0.8, label='U(1) Hypercharge Action')
    plt.plot(epochs, history['higgs'], 'm-.', alpha=0.8, label='Higgs Potential ($V(\phi)$)')
    plt.plot(epochs, history['fermion'], 'c:', linewidth=2, label='Fermion Action ($\overline{\psi} D \psi$)')
    
    plt.title("Vacuum Loss Landscape (Action Contributions over Time)")
    plt.xlabel("Cooling Steps (Epochs)")
    plt.ylabel("Action Energy")
    plt.yscale('log') # Log scale is crucial to see vastly different energy levels
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("plots/actions.png")


@torch.no_grad()
def measure_cornell_potential(model, r_max=6, t_fixed=4):
    """
    Measures the potential V(R) between static quarks.
    We look for the linear 'confinement' slope.
    """
    model.eval()
    u_su3 = get_gate(model.u_raw['su3'], model.partner_map, model.is_fwd)
    V_R = {}

    print(f"\n--- Measuring Cornell Potential (SU3) ---")
    for r in range(1, r_max + 1):
        # 1. Find all R x T loops on the lattice
        loop_idx = find_rectangular_loops(model.lattice_shape, model.edge_index, R=r, T=t_fixed)
        
        # 2. Gather and multiply gates along the loop
        u_p = u_su3[loop_idx] # [Num_Loops, 2R+2T, 3, 3]
        loop_prod = u_p[:, 0]
        for i in range(1, u_p.shape[1]):
            loop_prod = torch.matmul(loop_prod, u_p[:, i])
            
        # 3. W(R, T) = 1/3 * Re(Tr(Loop))
        # Averaged over all positions and orientations
        trace = torch.einsum('pii -> p', loop_prod).real / 3.0
        w_rt = trace.mean().item()
        
        # 4. Energy V(R) = - (1/T) * log(W(R,T))
        # Add a small epsilon to avoid log(0)
        v_r = - (1.0 / t_fixed) * np.log(max(w_rt, 1e-9))
        V_R[r] = v_r
        print(f"Distance R={r} | W(R,T)={w_rt:.4f} | V(R)={v_r:.4f}")
        
    return V_R


@torch.no_grad()
def measure_rho_parameter(model):
    """
    Checks the Electroweak consistency. 
    Rho = Mw^2 / (Mz^2 * cos^2_theta_W)
    """
    # 1. Extract Couplings from Betas
    # Standard Lattice mapping: Beta_SU2 = 4/g^2, Beta_U1 = 1/g'^2
    g_sq = 4.0 / model.betas['su2']
    gp_sq = 1.0 / model.betas['u1']
    
    # 2. Calculate Weinberg Angle (Tree Level)
    cos_sq_theta_w = g_sq / (g_sq + gp_sq)
    
    # 3. Measure the Higgs VEV from the simulation
    current_vev_sq = torch.sum(model.phi.abs()**2, dim=-1).mean().item()
    
    # 4. Calculate effective masses
    # These are lattice masses; the 'v' factor cancels out in the ratio
    mw_sq = 0.25 * g_sq * current_vev_sq
    mz_sq = 0.25 * (g_sq + gp_sq) * current_vev_sq
    
    rho = mw_sq / (mz_sq * cos_sq_theta_w)
    
    print(f"\n--- Electroweak Health Check ---")
    print(f"Weinberg Angle (cos²θw): {cos_sq_theta_w:.4f}")
    print(f"Measured Rho Parameter: {rho:.6f}")
    
    if abs(rho - 1.0) < 1e-3:
        print("✅ SUCCESS: The Higgs Mechanism is physically consistent.")
    else:
        print("⚠️ WARNING: Rho deviation detected. Check Higgs-Gauge coupling logic.")
        
    return rho


@torch.no_grad()
def measure_pion_mass(model, max_dist=10):
    """
    Measures the Pion mass by tracking the exponential decay 
    of the quark propagator from a central point source.
    """
    model.eval()
    u_su3 = get_gate(model.u_raw['su3'], model.partner_map, model.is_fwd)
    
    print(f"\n--- Measuring Pion Mass (Quark Propagator) ---")
    
    # 1. Create a Point Source at the center of the lattice
    # Shape: [Nodes, Color(3), Spin(4)]
    source = torch.zeros_like(model.pf_phi)
    
    # Find the center node index
    dims = model.lattice_shape
    center_coords = [d // 2 for d in dims]
    center_node = np.ravel_multi_index(center_coords, dims)
    
    # Inject 1.0 into all color/spin channels at the center
    source[center_node] = 1.0 + 0.0j
    
    # 2. Calculate the Propagator: x = (D^dag D)^-1 * source
    # This simulates the quark moving through the cooled SU(3) vacuum
    x = conjugate_gradient(
        model.fermion_calc.dirac, 
        source, 
        model.phi,
        model.edge_index, 
        model.edge_dirs, 
        u_su3, 
        model.is_fwd,
        max_iter=300, # Might need more iterations for a clean signal
        tol=1e-6
    )
    
    # 3. Calculate the Pion Correlator C(r)
    # The magnitude squared |x|^2 is proportional to the pion correlation
    correlator = torch.sum(x.abs()**2, dim=(-1, -2)) # Sum over color and spin
    
    # 4. Radially average the signal
    grid = np.indices(dims)
    
    # Calculate Manhattan distance (or Euclidean) from the center for all nodes
    # Dynamically create the reshape tuple based on the number of dimensions.
    # For 2D: (-1, 1, 1). For 3D: (-1, 1, 1, 1). For 4D: (-1, 1, 1, 1, 1).
    reshape_args = [-1] + [1] * len(dims)
    center_coords_array = np.array(center_coords).reshape(*reshape_args)
    
    # Calculate Manhattan distance (Taxicab) from the center for all nodes
    distances = np.sum(np.abs(grid - center_coords_array), axis=0).flatten()
    
    C_r = {}
    for r in range(1, max_dist + 1):
        mask = (distances == r)
        if mask.any():
            # Average the correlator for all nodes at distance r
            mean_signal = correlator[mask].mean().item()
            C_r[r] = mean_signal
            
            print(f"Distance r={r} | C(r)={mean_signal:.4e} | ln(C)={np.log(mean_signal):.4f}")
            
    return C_r


@torch.no_grad()
def measure_chiral_condensate(model, num_noise_vectors=5):
    """
    Measures Spontaneous Chiral Symmetry Breaking using a stochastic trace estimator.
    """
    model.eval()
    u_su3 = get_gate(model.u_raw['su3'], model.partner_map, model.is_fwd)
    dirac = model.fermion_calc.dirac
    
    print(f"\n--- Measuring Chiral Condensate ---")
    condensate_sum = 0.0
    
    for i in range(num_noise_vectors):
        # 1. Generate Z2 noise (random +/- 1 in complex plane)
        # This is more efficient for trace estimation than Gaussian noise
        noise = torch.randint(0, 2, model.pf_phi.shape, device=model.device).float() * 2 - 1
        noise = noise + 1j * (torch.randint(0, 2, model.pf_phi.shape, device=model.device).float() * 2 - 1)
        noise = noise / np.sqrt(2) # Normalize
        
        # 2. Solve x = (D^dag D)^-1 * noise
        x = conjugate_gradient(dirac, noise, model.phi, model.edge_index, model.edge_dirs, u_su3, model.is_fwd, max_iter=200)
        
        # 3. Apply D^dag to x to get (D^-1 * noise)
        # Recall: D^dag = gamma_5 * D * gamma_5
        g5_x = torch.einsum('ss, ecs -> ecs', dirac.g5, x)
        d_g5_x = dirac(g5_x, model.phi, model.edge_index, model.edge_dirs, u_su3, model.is_fwd)
        d_dag_x = torch.einsum('ss, ecs -> ecs', dirac.g5, d_g5_x)
        
        # 4. Compute eta^dag * D^-1 eta
        trace_est = torch.sum(noise.conj() * d_dag_x).real
        condensate_sum += trace_est.item()
        
    # Average over volume and noise vectors
    V = np.prod(model.lattice_shape)
    condensate = condensate_sum / (num_noise_vectors * V * 3 * 4) # Normalize by Volume, Color, Spin
    
    print(f"<ψ̄ψ> = {condensate:.6f}")
    return condensate


def run_gmor_test(model):
    """
    Tests the Gell-Mann-Oakes-Renner relation: m_pi^2 ∝ m_q
    """
    print("\n=== Starting GMOR Chiral Test ===")
    test_masses = [0.05, 0.10, 0.15, 0.20, 0.25]
    pion_masses = []
    
    for mq in test_masses:
        # Update the Dirac operator mass
        model.fermion_calc.dirac.bare_mass = mq
        
        # Measure the Pion correlator
        pion_signal = measure_pion_mass(model, max_dist=8)
        
        # Extract mass via linear fit of the log
        r_vals = list(pion_signal.keys())
        log_c = np.log(list(pion_signal.values()))
        
        # Fit from r=2 to r=6 to avoid edge noise
        slope, _ = np.polyfit(r_vals[1:6], log_c[1:6], 1)
        pion_masses.append(-slope)
        
    # Square the pion masses
    m_pi_sq = [m**2 for m in pion_masses]
    
    # Plotting
    plt.figure(figsize=(7,5))
    plt.plot(test_masses, m_pi_sq, 'bo-', label="$m_\pi^2$")
    
    # Fit line
    slope, intercept = np.polyfit(test_masses, m_pi_sq, 1)
    plt.plot(test_masses, [slope*x + intercept for x in test_masses], 'r--', 
             label=f"Fit: y = {slope:.2f}x + {intercept:.3f}")
    
    plt.xlabel("Input Quark Mass ($m_q$)")
    plt.ylabel("Pion Mass Squared ($m_\pi^2$)")
    plt.title("GMOR Relation Verification")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/gmor.png")


if __name__ == "__main__":
    # L_SHAPE = (16, 16, 16, 4)
    # L_SHAPE = (16, 16, 4)
    L_SHAPE = (16, 16)

    # model = VacuumFinder(
    #     lattice_shape=L_SHAPE,
    #     groups={'su3': 3, 'su2': 2, 'u1': 1},
    #     betas={'su3': 6.0, 'su2': 4.0, 'u1': 5.0} 
    # )
    
    model = QuantumVacuumFinder(
        lattice_shape=L_SHAPE,
        groups={'su3': 3, 'su2': 2, 'u1': 1},
        betas={'su3': 6.0, 'su2': 4.0, 'u1': 5.0} 
    )
    

    history = model.find_vacuum(steps=1000)

    plot_action_landscape(history)

    # 1. Check Electroweak Consistency
    rho = measure_rho_parameter(model)

    # 2. Check Strong Force Confinement
    potential = measure_cornell_potential(model, r_max=5)

    # 3. Visualize the Potential
    r_values = list(potential.keys())
    v_values = list(potential.values())
    
    plt.figure(figsize=(8,5))
    plt.plot(r_values, v_values, 'o-', label="Simulated Potential")
    
    # Fit a straight line to see the 'String Tension'
    slope, intercept = np.polyfit(r_values[1:], v_values[1:], 1)
    plt.plot(r_values, [slope*x + intercept for x in r_values], '--', 
             label=f"String Tension (σ) ≈ {slope:.3f}")
    
    plt.xlabel("Distance R (Lattice Units)")
    plt.ylabel("Potential V(R)")
    plt.title("SU(3) Quark Confinement (Cornell Potential)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/containement.png")


    pion_signal = measure_pion_mass(model, max_dist=12)

    # 5. Visualize the Exponential Decay
    r_vals = list(pion_signal.keys())
    c_vals = list(pion_signal.values())
    log_c_vals = np.log(c_vals)
    
    plt.figure(figsize=(8,5))
    plt.plot(r_vals, log_c_vals, 'o-', color='red', label="ln(Correlator)")
    
    # Fit a straight line to extract the mass (slope)
    # We ignore the first few points (r=1, 2) to avoid short-range lattice artifacts
    fit_start = 2
    slope, intercept = np.polyfit(r_vals[fit_start:], log_c_vals[fit_start:], 1)
    
    # The slope is -mass
    pion_mass = -slope
    
    plt.plot(r_vals, [slope*x + intercept for x in r_vals], 'k--', 
             label=f"Fit Mass (m_π) ≈ {pion_mass:.3f}")
    
    plt.xlabel("Distance r (Lattice Units)")
    plt.ylabel("ln(C(r))")
    plt.title("Pion Mass Extraction (Quark Propagator Decay)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/pion_mass.png")

    measure_chiral_condensate(model)
    run_gmor_test(model)





class WilsonDiracOperator2(nn.Module):
    def __init__(self, mass=0.1, r=1.0):
        super().__init__()
        self.mass = mass
        self.r = r
        
        # 1. Precompute Gamma matrices and register as buffers
        g1, g2, g3, g4 = get_gamma_matrices()
        self.register_buffer('g1', g1)
        self.register_buffer('g2', g2)
        self.register_buffer('g3', g3)
        self.register_buffer('g4', g4)
        
        # gamma_5 is required to calculate the exact Adjoint D^dag
        g5 = g1 @ g2 @ g3 @ g4
        self.register_buffer('g5', g5)
        self.register_buffer('id_spin', torch.eye(4, dtype=torch.complex64))

    def forward(self, psi, edge_index, edge_dirs, u_su3, is_fwd):
        dim_space = len(torch.unique(edge_dirs))
        out = (self.mass + self.r * dim_space) * psi
        
        src, dst = edge_index[0], edge_index[1]
        psi_j = psi[src]
        
        # Collect buffers into a list for easy indexing
        gammas = [self.g1, self.g2, self.g3, self.g4]
        
        for mu in range(dim_space):
            mask = (edge_dirs == mu)
            if not mask.any(): continue
            
            curr_src, curr_dst = src[mask], dst[mask]
            curr_u, curr_fwd = u_su3[mask], is_fwd[mask]
            
            rotated_psi = torch.einsum('ecd, eds -> ecs', curr_u, psi[curr_src])
            g = gammas[mu]
            
            if curr_fwd.any():
                proj_fwd = self.r * self.id_spin - g
                term_fwd = torch.einsum('ss, ecs -> ecs', proj_fwd, rotated_psi[curr_fwd])
                out.index_add_(0, curr_dst[curr_fwd], -0.5 * term_fwd)
                
            bwd_mask = ~curr_fwd
            if bwd_mask.any():
                proj_bwd = self.r * self.id_spin + g
                term_bwd = torch.einsum('ss, ecs -> ecs', proj_bwd, rotated_psi[bwd_mask])
                out.index_add_(0, curr_dst[bwd_mask], -0.5 * term_bwd)
                
        return out


def real_dot_product(a, b):
    # View complex tensors [N, ...] as real vectors [N, ..., 2]
    # Summing (Re(a)*Re(b) + Im(a)*Im(b)) is the same as Re(a* * b)
    a_real = torch.view_as_real(a)
    b_real = torch.view_as_real(b)
    return torch.sum(a_real * b_real)



class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # We use two real linear layers to represent W = A + iB
        self.fc_real = nn.Linear(in_features, out_features, bias=False)
        self.fc_imag = nn.Linear(in_features, out_features, bias=False)
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Complex initialization: 
        # Var(W) = 1 / in_features. We split this between real and imag parts.
        std = (1.0 / (2.0 * self.fc_real.in_features))**0.5
        nn.init.normal_(self.fc_real.weight, std=std)
        nn.init.normal_(self.fc_imag.weight, std=std)

    def forward(self, z):
        """
        z: Complex tensor of shape [..., in_features]
        Returns: Complex tensor of shape [..., out_features]
        """
        # Re(Wz + b) = (A*x - B*y) + b_re
        # Im(Wz + b) = (A*y + B*x) + b_im
        re = self.fc_real(z.real) - self.fc_imag(z.imag)
        im = self.fc_real(z.imag) + self.fc_imag(z.real)
        
        if self.bias_real is not None:
            re = re + self.bias_real
            im = im + self.bias_imag
            
        return torch.complex(re, im)



class UniversalGateFactory(nn.Module):
    def __init__(self, raw_u, partner_map, is_fwd):
        super().__init__()
        self.raw_u = raw_u
        self.partner_map = partner_map
        self.is_fwd = is_fwd

    def get_gate(self):
        # Use .adjoint() instead of .conj().transpose()
        # and .resolve_conj() to flatten the memory layout
        h = (self.raw_u - self.raw_u.adjoint()).resolve_conj()
        u_all = torch.matrix_exp(h)
        
        u_fwd = u_all[self.is_fwd]
        # Resolve the conjugate here too
        u_bwd_calculated = u_fwd.adjoint().resolve_conj()
        
        u_final = torch.zeros_like(u_all)
        u_final[self.is_fwd] = u_fwd
        u_final[self.partner_map[self.is_fwd]] = u_bwd_calculated
        
        return u_final

    def get_gate2(self):
        # 1. Project all raw tensors to the Group
        # We use the Exponential Map: U = exp(i * (A + A_dag))
        # This ensures every matrix is Unitarity/Special Unitarity
        h = self.raw_u - self.raw_u.conj().transpose(-2, -1)
        u_all = torch.matrix_exp(h)
        
        # Enforce U(-mu, x + mu) = U(mu, x)^dagger
        # This ensures the 'Loop' is mathematically closed.
        u_fwd = u_all[self.is_fwd]
        u_bwd_calculated = u_fwd.conj().transpose(-2, -1)
        
        # Reconstruct the full edge tensor using the partner_map
        u_final = torch.zeros_like(u_all)
        u_final[self.is_fwd] = u_fwd
        u_final[self.partner_map[self.is_fwd]] = u_bwd_calculated
        
        return u_final


class UniversalGateFactory2:
    def __init__(self, raw_links, partner_map, is_forward):
        """
        raw_links: [Edges, G, G] - The raw (potentially unconstrained) tensors.
        partner_map: [Edges] - Mapping to inverse edges.
        is_forward: [Edges] - Mask for +mu directions.
        """
        self.partner_map = partner_map    
        self.is_forward = is_forward
        self.u = self._enforce_physics(raw_links)

    def _enforce_physics(self, u):
        # 1. Force Adjointness: U_rev = U_fwd.H
        u_adj_partners = u[self.partner_map].conj().transpose(-2, -1)
        u_consistent = torch.where(self.is_forward.view(-1, 1, 1), u, u_adj_partners)
        
        # 2. Project to Group (Keep it Unitary)
        if u_consistent.shape[-1] == 1:
            return u_consistent / (torch.abs(u_consistent) + 1e-8)
        else:
            U, S, Vh = torch.linalg.svd(u_consistent)
            return torch.matmul(U, Vh)

    def get_gate(self):
        return self.u



class SimpleHiggsAction(nn.Module):
    """Legacy single-gauge-group variant; renamed so it does not shadow HiggsAction."""
    def __init__(self, v=1.0, lam=0.1):
        super().__init__()
        self.v = v      # Vacuum Expectation Value
        self.lam = lam  # Coupling constant (Self-interaction)

    def forward(self, phi, edge_index, u_gate):
        """
        phi: [Nodes, Group_Dim]
        u_gate: [Edges, G, G]
        """
        # 1. Potential Energy: V(phi) = lambda * (|phi|^2 - v^2)^2
        phi_sq = torch.norm(phi, dim=-1)**2
        v_pot = self.lam * torch.sum((phi_sq - self.v**2)**2)
        
        # 2. Kinetic/Interaction Energy: |phi_i - U_ij * phi_j|^2
        phi_j = phi[edge_index[0]] # Source node
        phi_i = phi[edge_index[1]] # Target node
        
        # Transport phi_j to node i: U_ij @ phi_j
        transported_phi = torch.einsum('egk, ek -> eg', u_gate, phi_j)
        
        # Difference (Covariant Derivative)
        diff = torch.norm(phi_i - transported_phi, dim=-1)**2
        kinetic = torch.sum(diff)
        
        return kinetic + v_pot


class HiggsVacuumFinder:
    def __init__(self, lattice_shape, group_dim, v=1.0, lam=0.1, beta=1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.group_dim = group_dim
        self.lattice_shape = lattice_shape
        
        # 1. Setup Lattice Geometry
        edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
        self.edge_index = edge_index.to(self.device)
        self.partner_map = partner_map.to(self.device)
        self.is_fwd = is_fwd.to(self.device)
        
        # 2. Find Plaquettes for Wilson Action
        plaq_idx = find_rectangular_loops(lattice_shape, edge_index) #, edge_dirs, is_fwd)
        
        # 3. Define Actions
        self.wilson_calc = WilsonAction(plaq_idx.to(self.device), group_dim, beta=beta)
        self.higgs_calc = SimpleHiggsAction(v=v, lam=lam)

        # 4. Initialize Fields as Learnable Parameters
        num_nodes = np.prod(lattice_shape)
        num_edges = edge_index.size(1)
        
        # Random initial state (High energy "hot" universe)
        self.phi = nn.Parameter(torch.randn(num_nodes, group_dim, dtype=torch.complex64, device=self.device))
        self.u_raw = nn.Parameter(torch.randn(num_edges, group_dim, group_dim, dtype=torch.complex64, device=self.device))
        
        self.optimizer = optim.Adam([self.phi, self.u_raw], lr=0.01)

    def find_vacuum(self, steps=1000):
        history = {
            'step': [],
            'total_action': [],
            'wilson_action': [],
            'higgs_action': [],
            'vev': [] # Vacuum Expectation Value (Average length of phi)
        }
        
        print("Initializing Big Bang... Cooling the universe...")
        for step in range(steps):
            self.optimizer.zero_grad()
            
            # 1. Enforce physics on gauge links
            factory = UniversalGateFactory(self.u_raw, self.partner_map, self.is_fwd)
            u_gate = factory.get_gate()
            
            # 2. Calculate Actions
            s_w = self.wilson_calc(u_gate)
            s_h = self.higgs_calc(self.phi, self.edge_index, u_gate)
            s_total = s_w + s_h
            
            # 3. Calculate VEV (Average magnitude of the Higgs field)
            current_vev = torch.norm(self.phi, dim=-1).mean()
            
            # 4. Record History
            history['step'].append(step)
            history['total_action'].append(s_total.item())
            history['wilson_action'].append(s_w.item())
            history['higgs_action'].append(s_h.item())
            history['vev'].append(current_vev.item())
            
            # 5. Optimize
            s_total.backward()
            self.optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step:03d} | Total: {s_total.item():.2f} | "
                      f"Wilson: {s_w.item():.2f} | Higgs: {s_h.item():.2f} | "
                      f"VEV: {current_vev.item():.3f}")

        return self.phi.detach(), u_gate.detach(), history


class ElectroweakVacuumFinder:
    def __init__(self, lattice_shape, v=1.0, lam=0.1, beta_su2=1.0, beta_u1=1.0):
        self.device = torch.device('cpu') # Use 'cuda' if available
        self.lattice_shape = lattice_shape
        
        # Lattice Geometry
        edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
        self.edge_index = edge_index
        self.partner_map = partner_map
        self.is_fwd = is_fwd
        plaq_idx = find_rectangular_loops(lattice_shape, edge_index) #, edge_dirs, is_fwd)
        
        # Actions
        self.wilson_su2 = WilsonAction(plaq_idx, group_dim=2, beta=beta_su2)
        self.wilson_u1  = WilsonAction(plaq_idx, group_dim=1, beta=beta_u1)
        self.higgs_calc = UnifiedHiggsAction(v=v, lam=lam)

        # Fields
        num_nodes, num_edges = np.prod(lattice_shape), edge_index.size(1)
        self.phi = nn.Parameter(torch.randn(num_nodes, 2, dtype=torch.complex64))
        self.u_su2_raw = nn.Parameter(torch.randn(num_edges, 2, 2, dtype=torch.complex64))
        self.u_u1_raw  = nn.Parameter(torch.randn(num_edges, 1, 1, dtype=torch.complex64))
        
        self.optimizer = optim.Adam([self.phi, self.u_su2_raw, self.u_u1_raw], lr=0.01)

    def find_vacuum(self, steps=1000):
        for step in range(steps):
            self.optimizer.zero_grad()
            
            # Enforce Physics (Adjointness/Unitarity) for both groups
            u_su2 = UniversalGateFactory(self.u_su2_raw, self.partner_map, self.is_fwd).get_gate()
            u_u1  = UniversalGateFactory(self.u_u1_raw, self.partner_map, self.is_fwd).get_gate()
            
            # Total Action = SU2 Energy + U1 Energy + Higgs Interaction
            loss = self.wilson_su2(u_su2) + self.wilson_u1(u_u1) + self.higgs_calc(self.phi, self.edge_index, u_su2, u_u1)
            
            loss.backward()
            self.optimizer.step()
            
        return self.phi.detach(), u_su2.detach(), u_u1.detach()


        
class WilsonDiracOperator2(nn.Module):
    def __init__(self, mass=0.1, r=1.0):
        super().__init__()
        self.mass = mass
        self.r = r # Wilson parameter (usually 1.0)

    def forward(self, psi, edge_index, edge_dirs, u_su3, is_fwd):
        """
        psi: [Nodes, Color(3), Spin(4)] - The Quark Field
        u_su3: [Edges, 3, 3] - The Gluon Field
        edge_dirs: [Edges] - Direction of the link (0, 1, 2, 3)
        """
        num_nodes = psi.shape[0]
        device = psi.device
        gammas = get_gamma_matrices(device)
        id_spin = torch.eye(4, dtype=torch.complex64, device=device)
        
        # 1. Diagonal Term: (m + 4r) * psi
        # In 3D lattice, use 3. In 4D, use 4.
        dim_space = len(torch.unique(edge_dirs))
        out = (self.mass + self.r * dim_space) * psi
        
        # 2. Hopping Terms
        src, dst = edge_index[0], edge_index[1]
        
        # Gather neighbors
        psi_j = psi[src] # Spinor at the other end of the link
        u_ij = u_su3    # Gluon link connecting them
        
        # Project Spin and Rotate Color
        # We need to calculate: (r*I -/+ gamma_mu) @ U @ psi_j
        res = torch.zeros_like(psi)
        
        for mu in range(dim_space):
            # Mask for edges pointing in direction mu
            mask = (edge_dirs == mu)
            if not mask.any(): continue
            
            curr_src = src[mask]
            curr_dst = dst[mask]
            curr_u = u_ij[mask]
            curr_psi = psi[curr_src]
            curr_fwd = is_fwd[mask]
            
            # Color rotation: U @ psi (acts on the color index 'c')
            # curr_u: [E, c, c], curr_psi: [E, c, s] -> [E, c, s]
            rotated_psi = torch.einsum('ecd, eds -> ecs', curr_u, curr_psi)
            
            # Spin projection: (r*I - gamma_mu) if forward, (r*I + gamma_mu) if backward
            # Note: Gamma matrices act on the spin index 's'
            g = gammas[mu]
            
            # Forward links
            fwd_mask = curr_fwd
            if fwd_mask.any():
                proj_fwd = self.r * id_spin - g
                term_fwd = torch.einsum('ss, ecs -> ecs', proj_fwd, rotated_psi[fwd_mask])
                out.index_add_(0, curr_dst[fwd_mask], -0.5 * term_fwd)
                
            # Backward links
            bwd_mask = ~curr_fwd
            if bwd_mask.any():
                proj_bwd = self.r * id_spin + g
                term_bwd = torch.einsum('ss, ecs -> ecs', proj_bwd, rotated_psi[bwd_mask])
                out.index_add_(0, curr_dst[bwd_mask], -0.5 * term_bwd)
                
        return out


class PseudofermionAction2(nn.Module):
    def __init__(self, mass=0.1):
        super().__init__()
        self.dirac = WilsonDiracOperator(mass=mass)

    def forward(self, pf_phi, edge_index, edge_dirs, u_su3, is_fwd):
        # We need to compute pf_phi^dag * (D^dag D)^-1 * pf_phi
        # 1. Solve (D^dag D) x = pf_phi
        # In a real training loop, we'd use 'detach' or special gradients, 
        # but for a Vacuum Finder, we want to see how U_su3 reacts.
        sol = conjugate_gradient(self.dirac, pf_phi, edge_index, edge_dirs, u_su3, is_fwd)
        
        # 2. Action value
        action = torch.sum(pf_phi.conj() * sol).real
        return action


def align_vacuum_to_unitary_gauge(phi_vac, u_su2_vac, edge_index):
    """
    Transforms the system into the Unitary Gauge where phi = [0, v].
    """
    num_nodes = phi_vac.shape[0]
    device = phi_vac.device
    
    # 1. Generate the local rotation G(x) for every node
    g_mats = torch.zeros((num_nodes, 2, 2), dtype=torch.complex64, device=device)
    phi_aligned = torch.zeros_like(phi_vac)
    
    for n in range(num_nodes):
        p = phi_vac[n]
        norm = torch.norm(p)
        # We need a matrix G such that G @ p = [0, norm]
        # In SU(2), for p = [a, b], G = [[b, -a], [conj(a), conj(b)]] / norm
        a, b = p[0], p[1]
        denom = norm + 1e-10
        g = torch.tensor([[b, -a],
                          [a.conj(), b.conj()]], dtype=torch.complex64, device=device) / denom
        g_mats[n] = g
        phi_aligned[n] = torch.tensor([0, norm], dtype=torch.complex64, device=device)

    # 2. Update the SU(2) Gauge Links
    # U'_{ij} = G_i * U_{ij} * G_j^H
    g_src = g_mats[edge_index[0]] # [Edges, 2, 2]
    g_tgt = g_mats[edge_index[1]] # [Edges, 2, 2]
    
    # Batch matrix multiplication: G_tgt @ U_su2 @ G_src.H
    u_su2_aligned = torch.einsum('eij, ejk, elk -> eil', g_tgt, u_su2_vac, g_src.conj())
    
    return phi_aligned, u_su2_aligned


def measure_boson_mass(phi_vac, u_vac, edge_index, higgs_calc, group_dim=2, eps=1e-3):
    print("\n--- Measuring Gauge Boson Mass ---")
    device = u_vac.device
    num_edges = u_vac.shape[0]
    
    # 1. Base Energy of the Vacuum
    with torch.no_grad():
        s_base = higgs_calc(phi_vac, edge_index, u_vac).item()

    # 2. Create a tiny Gauge Excitation (W-Boson)
    # For SU(2), we use the first Pauli matrix (sigma_1) to generate a rotation
    if group_dim == 2:
        tau_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    elif group_dim == 1:
        tau_1 = torch.ones((1, 1), dtype=torch.complex64, device=device)
    else: # SU(3) - Gell-Mann matrix 1
        tau_1 = torch.zeros((3, 3), dtype=torch.complex64, device=device)
        tau_1[0, 1] = 1; tau_1[1, 0] = 1

    # U_eps = exp(i * eps * tau_1)
    u_eps = torch.matrix_exp(1j * eps * tau_1)
    
    # 3. Apply the excitation to all links in the universe
    # We multiply the perturbation onto the existing vacuum links
    u_perturbed = torch.einsum('ij, ejk -> eik', u_eps, u_vac)
    
    # 4. Measure the Energy "Pushback"
    with torch.no_grad():
        s_new = higgs_calc(phi_vac, edge_index, u_perturbed).item()
        
    delta_s = s_new - s_base
    
    # The energy is summed over all edges, so we find the energy cost PER EDGE
    delta_s_per_edge = delta_s / num_edges
    
    # 5. Extract Mass: Delta S = 1/2 * M^2 * eps^2  => M = sqrt(2 * Delta S / eps^2)
    # (Note: We add max(0, ...) to prevent floating point negative zeros)
    mass = np.sqrt(max(0, 2 * delta_s_per_edge) / (eps**2))
    
    print(f"Energy Cost of Perturbation: {delta_s_per_edge:.2e}")
    print(f"Calculated Boson Mass:       {mass:.4f}")
    
    return mass


def measure_electroweak_masses(phi_vac, u_su2_vac, u_u1_vac, edge_index, higgs_calc, g_su2=1.0, g_u1=0.5, eps=1e-3):
    print("\n--- Measuring Electroweak Boson Masses ---")
    device = phi_vac.device
    num_edges = u_su2_vac.shape[0]
    
    # Create a 1x1 Identity for the U(1) slot when we want to bypass it
    # because our perturbation is already applied to the combined SU(2) space
    u1_identity = torch.ones((num_edges, 1, 1), dtype=torch.complex64, device=device)

    # 1. Base Energy
    with torch.no_grad():
        # Call with both vacuum fields
        s_base = higgs_calc(phi_vac, edge_index, u_su2_vac, u_u1_vac).item()

    # 2. Generators
    tau_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    tau_3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    I_su2 = torch.eye(2, dtype=torch.complex64, device=device)

    # 3. The Weinberg Angle
    theta_w = np.arctan(g_u1 / g_su2)
    print(f"Weinberg Angle (theta_W): {np.degrees(theta_w):.2f}°")

    # 4. Define the Excitations (The 'Kicks')
    # 1. W Boson: Pure SU(2) excitation
    gen_W = tau_1 
    
    # 2. The Unbroken Generator (The Photon A)
    # The Photon is the path where the g-coupling and g'-coupling cancel out.
    # For a Higgs at [0, v], this is the combination that satisfies Q|phi> = 0.
    # In this lattice setup, it is simply the sum of tau_3 and Identity.
    gen_A = (tau_3 + I_su2) 

    # 3. The Broken Generator (The Z Boson)
    # The Z is the combination 'orthogonal' to the photon in the (g, g') space.
    # This is the direction that the Higgs 'hates' the most.
    gen_Z = (g_su2**2 * tau_3 - g_u1**2 * I_su2) / (g_su2**2 + g_u1**2)**0.5
    
    # # W Boson: Excitation in the tau_1 direction (SU2)
    # gen_W = tau_1 
    
    # # Z Boson: Mixed excitation
    # # Since Higgs has Y=1, the U(1) part is effectively a phase on the whole SU(2) doublet
    # gen_Z = np.cos(theta_w) * tau_3 - np.sin(theta_w) * I_su2
    
    # # Photon: The unbroken combination
    # gen_A = np.sin(theta_w) * tau_3 + np.cos(theta_w) * I_su2

    masses = {}
    
    # Combine vacuum fields into a single starting point for perturbation
    u_comb_vac = u_su2_vac * u_u1_vac 

    for name, generator in [("W Boson", gen_W), ("Z Boson", gen_Z), ("Photon", gen_A)]:
        u_su2_pure = torch.eye(2, dtype=torch.complex64, device=device).repeat(num_edges, 1, 1)
        u_u1_pure  = torch.ones((num_edges, 1, 1), dtype=torch.complex64, device=device)
        
        # 2. Base Energy of the Clean Vacuum
        with torch.no_grad():
            s_base = higgs_calc(phi_vac, edge_index, u_su2_pure, u_u1_pure).item()

        # 3. Apply the perturbation to the Clean Vacuum
        u_eps = torch.matrix_exp(1j * eps * generator)
        
        # Measure: We put the perturbation in the SU(2) slot and keep U1 as Identity
        with torch.no_grad():
            # u_eps is the perturbed SU(2) link
            s_new = higgs_calc(phi_vac, edge_index, u_eps, u_u1_pure).item()
                
        delta_s_per_edge = (s_new - s_base) / num_edges
        mass = np.sqrt(max(0, 2 * delta_s_per_edge) / (eps**2))
    
        # # Apply excitation to the combined vacuum link
        # u_eps = torch.matrix_exp(1j * eps * generator)
        # u_comb_perturbed = torch.einsum('ij, ejk -> eik', u_eps, u_comb_vac)
        
        # # Measure energy using the 4-argument signature
        # # We pass u_comb_perturbed as the 'u_su2' and Identity as 'u_u1'
        # with torch.no_grad():
        #     s_new = higgs_calc(phi_vac, edge_index, u_comb_perturbed, u1_identity).item()
            
        # delta_s_per_edge = (s_new - s_base) / num_edges
        # mass = np.sqrt(max(0, 2 * delta_s_per_edge) / (eps**2))
        masses[name] = mass
        
        print(f"{name:8} Mass: {mass:.4f}")
        
    return masses



def apply_gauge_transformation(psi, u, edge_index, group_dim):
    """
    Applies a random local rotation G(x) to the fields.
    psi: [Nodes, Channels, G]
    u: [Edges, G, G]
    """
    num_nodes = psi.shape[0]
    device = psi.device
    
    # 1. Generate G
    if group_dim == 1:
        g = torch.exp(1j * torch.randn(num_nodes, 1, 1, device=device) * 2 * np.pi)
    else:
        h = torch.randn(num_nodes, group_dim, group_dim, dtype=torch.complex64, device=device)
        h = h + h.conj().transpose(-2, -1)
        g = torch.matrix_exp(1j * h)

    # 2. Transform Matter: psi' = G * psi
    psi_prime = torch.einsum('ngh, nch -> ncg', g, psi)
    
    # 3. Transform Links: U'_{target, source} = G_target * U * G_source^H
    # edge_index[0] is source (j), edge_index[1] is target (i)
    g_src = g[edge_index[0]] 
    g_tgt = g[edge_index[1]] 
    
    # Target on the left, Source on the right
    # Calculation: G_tgt [i,j] * U [j,k] * G_src_conj_transposed [k,l]
    # In einsum: 'nij' (G_tgt), 'njk' (U), 'nlk' (G_src.conj())
    u_prime = torch.einsum('nij, njk, nlk -> nil', g_tgt, u, g_src.conj())
    return psi_prime, u_prime, g


# def compute_wilson_action(u_gate, edge_index, edge_dirs, is_fwd, link_map):
#     """
#     u_gate: [Edges, G, G]
#     link_map: From get_link_map
#     """
#     num_nodes, num_dims, _ = link_map.shape
#     group_dim = u_gate.shape[-1]
#     total_action = 0.0

#     # We iterate over all unique pairs of dimensions (mu < nu) to find planes
#     for mu in range(num_dims):
#         for nu in range(mu + 1, num_dims):
#             # 1. Get the four links that form the square at every node
#             # Link 1: x -> x + mu
#             idx1 = link_map[:, mu, 1] 
#             u1 = u_gate[idx1]
            
#             # Link 2: x + mu -> x + mu + nu
#             node_x_plus_mu = edge_index[1, idx1]
#             idx2 = link_map[node_x_plus_mu, nu, 1]
#             u2 = u_gate[idx2]
            
#             # Link 3: x + mu + nu -> x + nu (Moving backward in mu)
#             node_x_plus_mu_plus_nu = edge_index[1, idx2]
#             idx3 = link_map[node_x_plus_mu_plus_nu, mu, 0]
#             u3 = u_gate[idx3]
            
#             # Link 4: x + nu -> x (Moving backward in nu)
#             node_x_plus_nu = edge_index[1, idx3]
#             idx4 = link_map[node_x_plus_nu, nu, 0]
#             u4 = u_gate[idx4]
            
#             # 2. Compute the Plaquette: P = U1 * U2 * U3 * U4
#             # For U(1) [scalar], this is just multiplication. 
#             # For SU(N), it's matrix multiplication.
#             if group_dim == 1:
#                 plaquettes = u1 * u2 * u3 * u4
#             else:
#                 # Matrix multiply: (u1 @ u2) @ (u3 @ u4)
#                 plaquettes = torch.matmul(torch.matmul(u1, u2), torch.matmul(u3, u4))
            
#             # 3. Calculate Action: S = sum( 1 - 1/N * Re(Tr(P)) )
#             # Trace is the sum of diagonal elements
#             if group_dim == 1:
#                 trace = plaquettes.real
#             else:
#                 # Batch trace for SU(N)
#                 trace = torch.diagonal(plaquettes, dim1=-2, dim2=-1).sum(-1).real
            
#             # Normalized action per plaquette
#             total_action += torch.sum(1.0 - (trace / group_dim))

#     return total_action


def run_invariance_test(group_dim=1, lattice_shape=(4, 4), channels=8, eps=1e-4):
    device = torch.device('cpu')
    
    # 1. Setup Lattice and Model
    edge_index, _, is_fwd, partner_map = create_lattice(lattice_shape)
    conv = GaugeEquivariantConv(channels)
    
    # 2. Setup Random Fields
    num_nodes = np.prod(lattice_shape)
    psi = torch.randn(num_nodes, channels, group_dim, dtype=torch.complex64)
    u_raw = torch.randn(edge_index.size(1), group_dim, group_dim, dtype=torch.complex64)
    
    # Ensure u is physical (adjointness)
    factory = UniversalGateFactory(u_raw, partner_map, is_fwd)
    u = factory.get_gate()

    # 3. Original Forward Pass
    out_orig = conv(psi, edge_index, u)
    
    # 4. Apply Gauge Transformation
    psi_prime, u_prime, g = apply_gauge_transformation(psi, u, edge_index, group_dim)
    
    # 5. Transformed Forward Pass
    out_trans = conv(psi_prime, edge_index, u_prime)
    
    # --- VERIFICATION ---
    
    # A. Check EQUIVARIANCE: Does Out' == G * Out?
    # Rotate the original output to see if it matches the transformed output
    expected_out_trans = torch.einsum('nij, ncj -> nci', g, out_orig)
    equiv_error = torch.abs(out_trans - expected_out_trans).mean().item()

    # B. Check INVARANCE: Does |Out'|^2 == |Out|^2?
    inv_orig = torch.norm(out_orig, dim=-1)**2
    inv_trans = torch.norm(out_trans, dim=-1)**2
    inv_error = torch.abs(inv_orig - inv_trans).mean().item()

    print(f"\n--- Symmetry Test (Group Dim: {group_dim}, Lattice: {lattice_shape}) ---")
    print(f"Equivariance Error: {equiv_error:.2e}")
    print(f"Invariance Error:    {inv_error:.2e}")
    
    if inv_error < eps:
        print("✅ SUCCESS: The model preserves Gauge Symmetry.")
    else:
        print("❌ FAILURE: Symmetry broken.")


def run_action_invariance_test(group_dim=3, lattice_shape=(4, 4), eps=1e-4):
    edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
    plaq_idx = find_rectangular_loops(lattice_shape, edge_index) #, edge_dirs, is_fwd)
    
    # Calculate the actual number of nodes in the lattice
    num_nodes = np.prod(lattice_shape) # 64 for (8, 8)
    
    calc = WilsonAction(plaq_idx, group_dim)
    
    # 2. Random Gauge Field
    u_raw = torch.randn(edge_index.size(1), group_dim, group_dim, dtype=torch.complex64)
    factory = UniversalGateFactory(u_raw, partner_map, is_fwd)
    u = factory.get_gate()
    
    # 3. Calculate Original Action
    s_orig = calc(u)
    
    # 4. Corrected Transformation
    # Pass a dummy field with the CORRECT number of nodes
    dummy_psi = torch.randn(num_nodes, 1, group_dim, dtype=torch.complex64)
    _, u_prime, _ = apply_gauge_transformation(dummy_psi, u, edge_index, group_dim)
    
    s_trans = calc(u_prime)
    
    diff = torch.abs(s_orig - s_trans)
    print(f"\nAction Invariance Test (SU({group_dim})):")
    print(f"Original S: {s_orig.item():.4f}")
    print(f"Transformed S: {s_trans.item():.4f}")
    print(f"Difference: {diff.item():.2e}")
    
    if diff < eps:
        print("✅ SUCCESS: The model preserves the Wilson Action.")
    else:
        print("❌ FAILURE: The model do not preserves the Wilson Action.")


def test_cold_vacuum(group_dim=3, lattice_shape=(4, 4, 4)):
    print(f"\n--- Cold Vacuum Test (SU({group_dim})) ---")
    edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
    plaq_idx = find_rectangular_loops(lattice_shape, edge_index) #, edge_dirs, is_fwd)
    
    calc = WilsonAction(plaq_idx, group_dim)
    
    # All links = Identity
    u_identity = torch.eye(group_dim, dtype=torch.complex64).repeat(edge_index.size(1), 1, 1)
    
    action = calc(u_identity)
    print(f"Vacuum Action: {action.item():.6e}")
    
    if abs(action.item()) < 1e-7:
        print("✅ SUCCESS: Flat space has zero energy.")
    else:
        print("❌ FAILURE: Non-zero vacuum energy.")


def test_translation_invariance(group_dim=3, lattice_shape=(4, 4, 4)):
    print(f"\n--- Translation Invariance Test (SU({group_dim})) ---")
    edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
    plaq_idx = find_rectangular_loops(lattice_shape, edge_index) #, edge_dirs, is_fwd)
    calc = WilsonAction(plaq_idx, group_dim)

    u_raw = torch.randn(edge_index.size(1), group_dim, group_dim, dtype=torch.complex64)
    u = UniversalGateFactory(u_raw, partner_map, is_fwd).get_gate()

    num_nodes = np.prod(lattice_shape)
    num_edges = edge_index.size(1)

    # 1. Map: Node Index -> Coordinate tuple
    # Example: 5 -> (1, 0, 1)
    coords = np.indices(lattice_shape).reshape(len(lattice_shape), -1).T
    node_to_coord = {i: tuple(c) for i, c in enumerate(coords)}
    coord_to_node = {tuple(c): i for i, c in enumerate(coords)}

    # 2. Map: (Source_Node, Dir, Is_Fwd) -> Edge_Index
    # This identifies the physical "identity" of an edge
    edge_registry = {}
    for i in range(num_edges):
        src = edge_index[0, i].item()
        d = edge_dirs[i].item()
        f = is_fwd[i].item()
        edge_registry[(src, d, f)] = i

    # 3. Create the Shifted Gauge Field
    # Shift vector: [1, 0, 0, ...] (Move 1 step in the first dimension)
    shift = np.zeros(len(lattice_shape), dtype=int)
    shift[0] = 1 
    
    u_shifted = torch.zeros_like(u)

    for i in range(num_edges):
        # Current edge info
        src_node, d, f = edge_index[0, i].item(), edge_dirs[i].item(), is_fwd[i].item()
        src_coord = np.array(node_to_coord[src_node])
        
        # New source coordinate (with periodic wrap-around)
        new_src_coord = tuple((src_coord + shift) % np.array(lattice_shape))
        new_src_node = coord_to_node[new_src_coord]
        
        # Find the edge index that represents the same direction at the new node
        target_edge_idx = edge_registry[(new_src_node, d, f)]
        
        # Move the link variable
        u_shifted[target_edge_idx] = u[i]

    # 4. Compare Actions
    s_orig = calc(u)
    s_shifted = calc(u_shifted)
    
    diff = torch.abs(s_orig - s_shifted).item()
    print(f"Original Action:  {s_orig.item():.6f}")
    print(f"Shifted Action:   {s_shifted.item():.6f}")
    print(f"Difference:       {diff:.2e}")

    if diff < 1e-5:
        print("✅ SUCCESS: The Action is Translationally Invariant.")
    else:
        print("❌ FAILURE: The Action depends on the absolute lattice position.")


def test_lattice_homogeneity(group_dim=3, lattice_shape=(4, 4, 4)):
    print(f"\n--- Lattice Homogeneity Test ---")
    edge_index, edge_dirs, is_fwd, partner_map = create_lattice(L_SHAPE)
    plaq_idx = find_rectangular_loops(L_SHAPE, edge_index, edge_dirs, is_fwd)
    calc = WilsonAction(plaq_idx, group_dim=G)
    
    # Count how many times each edge appears in a plaquette
    edge_counts = torch.zeros(num_edges)
    edge_counts.scatter_add_(0, plaq_idx.flatten(), torch.ones(plaq_idx.numel()))
    
    unique_counts = torch.unique(edge_counts)
    print(f"Edges participate in {unique_counts.tolist()} plaquettes.")
    
    if len(unique_counts) == 1:
        print("✅ SUCCESS: Every part of the lattice is identical (Translation Invariant).")
    else:
        print("❌ FAILURE: The lattice has 'holes' or 'borders'.")
        

def test_elitzurs_theorem(group_dim=3, num_samples=2000):
    print(f"\n--- Elitzur's Theorem Test (SU({group_dim})) ---")
    links = []
    device = torch.device('cpu')
    
    for _ in range(num_samples):
        if group_dim == 1:
            u = torch.exp(1j * torch.randn(1, 1, device=device) * 2 * np.pi)
        else:
            # Proper Haar Measure sampling using QR decomposition
            z = (torch.randn(group_dim, group_dim, dtype=torch.complex64, device=device))
            q, r = torch.linalg.qr(z)
            
            # Extract the diagonal of R to fix the phase of Q
            d = torch.diagonal(r)
            phases = d / torch.abs(d)
            u = q @ torch.diag(phases)
            
            # Ensure det(U) = 1 for SU(N)
            det = torch.linalg.det(u)
            u = u / (det**(1/group_dim))
            
        links.append(u)
    
    avg_link = torch.stack(links).mean(dim=0)
    magnitude = torch.norm(avg_link).item()
    
    print(f"Average Link Magnitude: {magnitude:.4f}")
    # With 2000 samples, it should be very small (< 0.05)
    if magnitude < 0.05: 
        print("✅ SUCCESS: The universe has no preferred direction.")
    else:
        print("❌ FAILURE: Symmetry bias detected.")


def run_higgs_invariance_test(group_dim=2, lattice_shape=(4, 4)):
    print(f"\n--- Higgs Gauge Invariance Test (SU({group_dim})) ---")
    
    # 1. Setup Lattice
    edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
    num_nodes = np.prod(lattice_shape)
    
    # 2. Random Fields
    # Higgs lives on nodes: [Nodes, Group_Dim]
    phi = torch.randn(num_nodes, group_dim, dtype=torch.complex64)
    # Gauge links live on edges: [Edges, Group_Dim, Group_Dim]
    u_raw = torch.randn(edge_index.size(1), group_dim, group_dim, dtype=torch.complex64)
    u = UniversalGateFactory(u_raw, partner_map, is_fwd).get_gate()
    
    # 3. Calculate Original Higgs Action
    calc = SimpleHiggsAction(v=1.0, lam=0.5)
    s_orig = calc(phi, edge_index, u)
    
    # 4. Apply Gauge Transformation G(x)
    # Note: We pass phi as the 'psi' argument to transform it like matter
    phi_prime, u_prime, g = apply_gauge_transformation(phi.unsqueeze(1), u, edge_index, group_dim)
    phi_prime = phi_prime.squeeze(1) # Back to [Nodes, Group_Dim]
    
    s_trans = calc(phi_prime, edge_index, u_prime)
    
    diff = torch.abs(s_orig - s_trans).item()
    print(f"Original S_Higgs:    {s_orig.item():.6f}")
    print(f"Transformed S_Higgs: {s_trans.item():.6f}")
    print(f"Difference:          {diff:.2e}")
    
    if diff < 1e-4:
        print("✅ SUCCESS: The Higgs Action respects Gauge Symmetry.")
    else:
        print("❌ FAILURE: Symmetry broken in Higgs sector.")


    # # 1. Find the Unified Vacuum
    # # We pass beta proportional to 1/g^2
    # finder = ElectroweakVacuumFinder(
    #     lattice_shape=L_SHAPE, 
    #     v=1.0, 
    #     lam=0.5, 
    #     beta_su2=1.0/(G_SU2**2), 
    #     beta_u1=1.0/(G_U1**2)
    # )
        
    # print("Cooling the Unified Electroweak Vacuum...")
    # phi_vac, u_su2_vac, u_u1_vac = finder.find_vacuum(steps=400)

    # print("\nStraightening the vacuum (Unitary Gauge)...")
    # phi_aligned, u_su2_aligned = align_vacuum_to_unitary_gauge(
    #     phi_vac, u_su2_vac, finder.edge_index
    # )

    # masses = measure_electroweak_masses(
    #     phi_aligned, 
    #     u_su2_aligned, 
    #     u_u1_vac, 
    #     finder.edge_index, 
    #     finder.higgs_calc,
    #     g_su2=G_SU2,
    #     g_u1=G_U1
    # )

    # mw, mz, ma = masses["W Boson"], masses["Z Boson"], masses["Photon"]
    # print(f"\nPhoton 'Mass' (should be 0): {ma:.6f}")
    
    # # Check the Weinberg relation
    # cos_theta = G_SU2 / np.sqrt(G_SU2**2 + G_U1**2)
    # rho = (mw**2) / (mz**2 * cos_theta**2)
    
    # print(f"Rho Parameter (should be 1.0): {rho:.4f}")
        

# if __name__ == "__main__":
#     # --- Physical Parameters ---
#     G = 2                # SU(2) Gauge Theory
#     L_SHAPE = (4, 4, 4)  # 3D Lattice (64 nodes, 192 edges)
#     V_TARGET = 1.0       # Target Vacuum Expectation Value (VEV)
#     LAMBDA = 0.5         # Higgs self-coupling strength
#     BETA = 1.0           # Gauge coupling strength
#     STEPS = 400          # "Time" steps for cooling

#     # --- 1. Run the Simulation ---
#     finder = HiggsVacuumFinder(
#         lattice_shape=L_SHAPE, 
#         group_dim=G, 
#         v=V_TARGET, 
#         lam=LAMBDA, 
#         beta=BETA
#     )
    
#     phi_out, u_out, history = finder.find_vacuum(steps=STEPS)

#     mass_w = measure_boson_mass(
#         phi_out, 
#         u_out, 
#         finder.edge_index, 
#         finder.higgs_calc, 
#         group_dim=G
#     )
    
#     masses = measure_electroweak_masses(
#         phi_vac, u_su2_vac, u_u1_vac, edge_index, higgs_calc, g_su2=1.0, g_u1=0.5, eps=1e-3
#     ):

#     # --- 2. Plotting the Results ---
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#     fig.suptitle('Spontaneous Symmetry Breaking: Cooling to the Vacuum', fontsize=16)

#     # Plot A: The Energies (Actions)
#     ax1.plot(history['step'], history['total_action'], label='Total Action', color='black', linewidth=2)
#     ax1.plot(history['step'], history['wilson_action'], label='Wilson Action (Gauge)', color='blue', linestyle='--')
#     ax1.plot(history['step'], history['higgs_action'], label='Higgs Action', color='red', linestyle='--')
#     ax1.set_title("System Energy vs Optimization Steps")
#     ax1.set_xlabel("Optimization Step (Cooling)")
#     ax1.set_ylabel("Action (Energy)")
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()

#     # Plot B: The Vacuum Expectation Value (VEV)
#     ax2.plot(history['step'], history['vev'], label='Measured VEV |φ|', color='purple', linewidth=2)
#     ax2.axhline(y=V_TARGET, color='green', linestyle=':', linewidth=2, label=f'Target VEV (v={V_TARGET})')
#     ax2.set_title("Higgs Field Magnitude (VEV)")
#     ax2.set_xlabel("Optimization Step")
#     ax2.set_ylabel("Average Length of Higgs Field")
#     ax2.set_ylim(0, V_TARGET * 1.5)
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()

#     plt.tight_layout()
#     plt.savefig("higgs.png")

#     # --- 3. Final Physical State Verification ---
#     print("\n--- Final Vacuum State ---")
    
#     # Calculate final VEV distribution
#     phi_magnitudes = torch.norm(phi_out, dim=-1)
#     mean_vev = phi_magnitudes.mean().item()
#     std_vev = phi_magnitudes.std().item()
    
#     print(f"Target VEV:      {V_TARGET}")
#     print(f"Achieved VEV:    {mean_vev:.4f} ± {std_vev:.4f}")
    
#     if abs(mean_vev - V_TARGET) < 0.05 and std_vev < 0.05:
#         print("✅ SUCCESS: The Higgs field has stably condensed.")
#         print("   The Gauge fields are now locked and massive!")
#     else:
#         print("❌ FAILURE: The vacuum did not settle. Try adjusting learning rate or lambda.")


# if __name__ == "__main__":
#     # Test Parameters
#     shape_2d = (8, 8)
#     shape_3d = (4, 4, 4) # Smaller for speed, but fully 3D
    
#     print("==========================================")
#     print("   LATTICE GAUGE THEORY: SYMMETRY SUITE   ")
#     print("==========================================\n")

#     for g_dim in [1, 2, 3]:
#         if g_dim == 1: 
#             print(f"\n>>>> TESTING GROUP: U({g_dim}) <<<<")
#         else:
#             print(f"\n>>>> TESTING GROUP: SU({g_dim}) <<<<")
        
#         # 1. GNN Equivariance Tests (3D)
#         run_invariance_test(group_dim=g_dim, lattice_shape=shape_3d)
        
#         # 2. Action Invariance Tests (3D)
#         run_action_invariance_test(group_dim=g_dim, lattice_shape=shape_3d)

#         run_higgs_invariance_test(group_dim=g_dim, lattice_shape=shape_3d)


#     print("\n==========================================")
#     print("      DEEP PHYSICS DIAGNOSTICS            ")
#     print("==========================================")

#     # 3. Cold Vacuum Check
#     test_cold_vacuum(group_dim=3, lattice_shape=shape_3d)
    
#     # 4. Translation Check
#     test_translation_invariance(group_dim=2, lattice_shape=shape_3d)
    
#     # 5. Elitzur's Theorem Check
#     test_elitzurs_theorem(group_dim=3)

#     print("\n==========================================")
#     print("           ALL TESTS COMPLETE             ")
#     print("==========================================")



# if __name__ == "__main__":
#     # # Test U(1), SU(2), SU(3) in 2D
#     # run_invariance_test(group_dim=1, lattice_shape=(8, 8))
#     # run_invariance_test(group_dim=2, lattice_shape=(8, 8))
#     # run_invariance_test(group_dim=3, lattice_shape=(8, 8))

#     # run_action_invariance_test(group_dim=1, lattice_shape=(8, 8))
#     # run_action_invariance_test(group_dim=2, lattice_shape=(8, 8))
#     # run_action_invariance_test(group_dim=3, lattice_shape=(8, 8))

#     # # Test U(1), SU(2), SU(3) in 3D
#     run_invariance_test(group_dim=1, lattice_shape=(8, 8, 8))
#     run_invariance_test(group_dim=2, lattice_shape=(8, 8, 8))
#     run_invariance_test(group_dim=3, lattice_shape=(8, 8, 8))

#     run_action_invariance_test(group_dim=1, lattice_shape=(8, 8, 8))
#     run_action_invariance_test(group_dim=2, lattice_shape=(8, 8, 8))
#     run_action_invariance_test(group_dim=3, lattice_shape=(8, 8, 8))

#     test_cold_vacuum()
#     test_translation_invariance()
#     test_elitzurs_theorem()

#     # # Test U(1), SU(2), SU(3) in 4D
#     # run_invariance_test(group_dim=1, lattice_shape=(8, 8, 8, 8))
#     # run_invariance_test(group_dim=2, lattice_shape=(8, 8, 8, 8))
#     # run_invariance_test(group_dim=3, lattice_shape=(8, 8, 8, 8))

