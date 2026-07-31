from dataclasses import dataclass, field

import yaml


@dataclass
class LatticeConfig:
    """Simulation Grid and Abstract Mass Parameters"""
    L: int = 16
    dims: int = 4
    hidden_dim: int = 64
    a_spacing: float = 0.125 # fm
    kappa: float = 0.01      # Kinetic hopping parameter
    m_sq_su3: float = 0.5    # Restoring mass to prevent color condensate
    m_sq_u1: float = 0.1     # Restoring mass for hypercharge
    v_target: float = 1.0    # Lattice units target for the VEV

@dataclass
class FermionMasses:
    """Fermion masses in GeV"""
    m_e: float = 0.000511
    m_mu: float = 0.1057
    m_tau: float = 1.78
    m_u: float = 0.0019
    m_d: float = 0.0044
    m_s: float = 0.087
    m_c: float = 1.32
    m_b: float = 4.24
    m_t: float = 173.5

@dataclass
class MixingParameters:
    """CKM Matrix and CP Violation"""
    theta_12: float = 13.1  # degrees
    theta_23: float = 2.4   # degrees
    theta_13: float = 0.2   # degrees
    delta_cp: float = 0.995 # radians
    theta_qcd: float = 0.0

@dataclass
class GaugeCouplings:
    """Gauge sector couplings at the Z-pole"""
    g1: float = 0.357 # U(1)
    g2: float = 0.652 # SU(2)
    g3: float = 1.221 # SU(3)

    @property
    def betas(self):
        """Translates physical 'g' to lattice 'beta' for the loss function."""
        return {
            'u1':  2.0 / (self.g1 ** 2),
            'su2': 4.0 / (self.g2 ** 2),
            'su3': 6.0 / (self.g3 ** 2)
        }

@dataclass
class HiggsSector:
    """Symmetry Breaking Parameters in GeV"""
    v_phys: float = 246.0   # GeV
    m_H: float = 125.09     # GeV

    @property
    def lambda_coupling(self):
        """Calculates physical lambda: lambda = m_H^2 / (2 * v^2)"""
        return (self.m_H ** 2) / (2 * (self.v_phys ** 2))

@dataclass
class TrainingConfig:
    epochs: int = 5000
    burn_in: int = 500
    
    # Neural Network Learning (Adam)
    lr_nn: float = 0.005
    
    # Physics/Gauge Learning (SGLD)
    lr_su3: float = 0.0001  # Strong force is "stiff," needs lower LR
    lr_su2: float = 0.0005
    lr_u1: float  = 0.001   # Hypercharge can usually handle higher LR
    
    temperature: float = 0.002
    sample_interval: int = 10
    check_interval: int = 100


@dataclass
class SMConfig:
    """Master Configuration Object"""
    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    fermions: FermionMasses = field(default_factory=FermionMasses)
    mixing: MixingParameters = field(default_factory=MixingParameters)
    gauge: GaugeCouplings = field(default_factory=GaugeCouplings)
    higgs: HiggsSector = field(default_factory=HiggsSector)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str):
        """Loads a nested YAML file and parses it into the dataclasses."""
        with open(path) as f:
            raw = yaml.safe_load(f)
            
        return cls(
            lattice=LatticeConfig(**raw.get('lattice', {})),
            fermions=FermionMasses(**raw.get('fermions', {})),
            mixing=MixingParameters(**raw.get('mixing', {})),
            gauge=GaugeCouplings(**raw.get('gauge', {})),
            higgs=HiggsSector(**raw.get('higgs', {})),
            training=TrainingConfig(**raw.get('training', {}))
        )