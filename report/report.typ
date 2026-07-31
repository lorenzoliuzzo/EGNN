#set text(size: 11pt)
#set page(margin: 1in, numbering: "1")
#set heading(numbering: "1.1")

#align(center)[
  #block(text(weight: "bold", 2em)[Simulation of $"SU"(N)$ Gauge Theories via Equivariant GNNs and Normalizing Flows])
  #v(1em)
  #text(style: "italic")[Updated Mathematical and Physical Technical Report]
]

= Introduction
This report details a computational framework for simulating lattice gauge theories using Graph Neural Networks (GNNs). The code implements a discrete version of the Standard Model's gauge group structure: $G = "SU"(3) times "SU"(2) times U(1)$. By utilizing gauge-equivariant message passing, the model preserves the fundamental symmetries of particle physics while leveraging Stochastic Gradient Langevin Dynamics (SGLD) for quantum ensemble sampling.

= Mathematical Framework

== Gauge Group Generators
The model defines the Lie algebra generators for the three components of the Standard Model. These matrices form the basis for the gauge fields $A_mu$:

- *SU(2) (Weak Interaction):* Represented by the three Pauli matrices $sigma_a$.
- *SU(3) (Strong Interaction):* Represented by the eight Gell-Mann matrices $lambda_a$.
- *U(1) (Electromagnetism/Hypercharge):* Represented by complex phase rotations.

The gauge links $U_(i j)$ are generated via the exponential map from the algebra to the group:
$ U_(i j) = exp(i sum_a alpha_a T^a) $
where $T^a$ are the generators and $alpha_a$ are the learned parameters (gauge potentials).

== Complex Activations
The core linear transformation in the `ComplexLinear` layer has been upgraded from a purely linear to an affine transformation by adding a complex bias $b = b_"re" + i b_"im"$:
$ f(z) = W z + b $
This allows the model to learn global field offsets, which is critical for identifying the specific vacuum state during symmetry breaking.

== Learnable ModReLU Activation
To handle complex-valued matter fields $phi in CC^n$, the code implements the `mod_relu` activation function:
$ "mod_relu"(z) = cases( (|z| + b) z/(|z|) & "if" |z| + b > 0, 0 & "otherwise" ) $
By making the bias $b$ a learnable parameter per channel, the network can dynamically adjust the "dead zone" magnitude where fluctuations are suppressed.


= Physics of the Architecture

== Parallel Transport and Covariant Convolution
In a gauge theory, comparing field values at different points ($phi_i$ and $phi_j$) is physically meaningless without a connection. The `SUN_GaugeConv` layer implements *Parallel Transport*. 

When a message is passed from node $j$ to node $i$, it is multiplied by the gauge link $U_(i j)$:
$ phi'_i = sum_(j in cal(N)(i)) U_(i j) phi_j $
This ensures that the output is *Gauge Equivariant*. If we perform a local gauge transformation $phi_i arrow.r g_i phi_i$, the link transforms as $U_(i j) arrow.r g_i U_(i j) g_j^dagger$, maintaining the consistency of the field interactions.

== The Action (Loss Function)
The training objective is the minimization of the Euclidean Action $S[phi, U]$, consisting of three primary terms:

+ *Potential Term ($S_V$):* Implements the "Mexican Hat" potential for Spontaneous Symmetry Breaking (SSB):
  $ S_V = lambda (sum_i |phi_i|^2 - v^2)^2 $
  This drives the field to a Vacuum Expectation Value (VEV) of $v$.
  
+ *Kinetic Term ($S_K$):* The discrete covariant derivative:
  $ S_K = kappa sum_(chevron.l i,j chevron.r) |phi_i - U_(i j) phi_j|^2 $
  This penalizes sharp variations in the field that are not aligned with the gauge field.

+ *Wilson Plaquette Action ($S_W$):* Represents the pure gauge field energy:
  $ S_W = beta sum_(square) (1 - 1/N "Re" "Tr"[U_1 U_2 U_3 U_4]) $
  where the sum is over all elementary squares (plaquettes) on the lattice.


= Sampling and Optimization

== Burn-in and Thermalization
The simulation uses the *Adam* optimizer for an initial "burn-in" phase. This is physically equivalent to *Thermalization*, where the system is driven from a random high-energy state toward the local minimum of the action (the classical configuration).

== Stochastic Gradient Langevin Dynamics (SGLD)
After burn-in, the code transitions to SGLD. SGLD adds Gaussian noise to the gradients:
$ theta_(t+1) = theta_t - epsilon nabla S(theta_t) + sqrt(2 epsilon T) eta_t $
where $eta_t tilde cal(N)(0,1)$. This allows the model to sample from the Boltzmann distribution $P(phi) prop exp(-S/T)$, effectively simulating *Quantum Fluctuations* around the vacuum.

== Gauge-Equivariant Convolution with Normalization
The `SUN_GaugeConv` layer performs parallel transport of fields across edges. To prevent runaway weight scaling, a normalization step is applied after the linear transformation:
$ phi'_i = (sum U_(i j) phi_j) / (chevron.l |phi| chevron.r + epsilon) $
This ensures that the GNN learns the spatial correlations of the field without being biased by its absolute magnitude.

== Jackknife Error Estimation
For the mass gap calculation, the model employs *Jackknife Resampling*. This provides a robust estimate of statistical errors for the effective mass $m_"eff" (t)$:
$ sigma_m (t) = sqrt((N - 1) / N sum_(i=1)^N (hat(m)_i (t) - bar(m)(t))^2) $
where $hat(m)_i$ is the mass calculated from the $i$-th jackknife block. This method is essential for identifying stable mass plateaus amidst quantum noise.


= Physical Observables

== Vacuum Expectation Value (VEV)
The model tracks the average magnitude of the Higgs-like field:
$ v = chevron.l |phi| chevron.r $
A stable, non-zero VEV indicates that the $"SU"(2)$ symmetry has been successfully broken.

== Mass Gap Calculation
The mass of the scalar particle is extracted from the exponential decay of the two-point correlation function $C(t)$:
$ C(t) = chevron.l phi(0) phi(t) chevron.r tilde exp(-m t) $
The code utilizes the `fsolve` method to find the effective mass $m$ using the cosh-ratio method, which accounts for the periodic boundary conditions on a finite lattice:
$ C(t)/C(t+1) = cosh(m(t - L/2)) / cosh(m(t + 1 - L/2)) $

= Conclusion
The provided implementation bridges the gap between Deep Learning and Lattice Field Theory. By embedding the $"SU"(N)$ group structure directly into the GNN architecture, it provides a robust platform for studying phase transitions, symmetry breaking, and particle masses in a gauge-invariant manner.