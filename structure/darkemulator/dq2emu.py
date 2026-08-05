"""Thin wrapper around Dark Emulator 2, giving it the (z, k) table interface
that dq2_interface_extrapolate.py expects.

NOTE (2026-07-30): the method names below were `get_lin_pk_tot` / `get_pk_tot`,
which do not exist in dark_emulator2 1.0.1 -- it spells them `*_total`.  Check
these if a future DE2 release moves them again; everything else in this file is
plain numpy.  `_total` is the right choice over the `get_pk` default of
pk_type="cb": CosmoSIS's matter_power_* means total matter (the cb spectrum
differs by ~1% at k = 0.1 h/Mpc for Mnu = 0.06 eV).
"""
#import sys
#sys.path.insert(0,'/lustre/work/terasawa/baccoemu/')
from dark_emulator2 import DarkEmulator2 as dq2
import numpy as np
from typing import Union
from dataclasses import dataclass
import copy


@dataclass(frozen=True)
class GeoConfig:
    zmax: Union[float, int] = 4.0
    nz: int = 401

@dataclass(frozen=True)
class PsConfig(GeoConfig):
    # CAUTION: DE2 itself only covers k = 1e-3 .. 1e2 h/Mpc. The defaults here
    # (1e-4 .. 10^1.5) reach outside that on the low-k side, where klist is
    # served by DE2's own interpolation rather than by the emulator. The
    # CosmoSIS interface passes kmin/kmax explicitly and stays inside the
    # native range; anything wider should be handled by the interface's own
    # linear_extend() so that the extrapolated region is at least explicit.
    logkh_min: Union[float, int] = -4
    logkh_max: Union[float, int] = 1.5
    logkh_min_lin: Union[float, int] = -4
    logkh_max_lin: Union[float, int] = 2
    nk: int = 700
    nk_lin: int = 700
    npart: float = 3000
    lbox: float = 1000
    
def window_tophat(kR):
    return np.float64(3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3.0)

class DQ2Emu:
    def __init__(self, config):
        if not isinstance(config, PsConfig):
            raise TypeError("input config is not a power spectrum configuration")
        self.nk = config.nk
        self.nz = config.nz
        
        self.ks = np.logspace(
            config.logkh_min,
            config.logkh_max,
            config.nk,
        )  # [h/Mpc]

        self.ks_lin = np.logspace(
            config.logkh_min_lin,
            config.logkh_max_lin,
            config.nk_lin,
        )  # [h/Mpc]

        
        self.z = np.linspace(
            0.0,
            config.zmax,
            config.nz,
        )
        self.a = 1.0 / (1 + self.z)
        # Nonlinear DQ2 emu 
        self.pk = dq2(npart=config.npart, lbox=config.lbox)
        return
    
    def get_sigma8(self, params):
        sigma8 = self.pk.lin_pk_emu.get_sigma8(params)
        
        return sigma8
    
    def compute_pk_table(self, params):
        """Computes the nonlinear power spectrum table.
        rows of the table: z; columns of the table: k
        """
        #pk_lin_table = np.zeros((self.nz, self.nk + 50))
        #pk_nl_table = np.zeros((self.nz, self.nk))
        _, pk_lin_table = self.pk.get_lin_pk_total(params, klist=self.ks,
                                           zred=self.z)
            
        _, pk_nl_table = self.pk.get_pk_total(params, klist=self.ks,
                                           zred=self.z)
           
                                    
        return pk_lin_table, pk_nl_table
    
    def compute_pnl_table(self, params):
        """Computes the nonlinear power spectrum table.
        rows of the table: z; columns of the table: k
        """
        _, pk_nl_table = self.pk.get_pk_total(params, klist=self.ks,
                                           zred=self.z)
           
                                    
        return pk_nl_table
    
    def compute_plin_table(self, params):
        """Computes the linear power spectrum table.
        rows of the table: z; columns of the table: k
        """
        _, pk_lin_table = self.pk.get_lin_pk_total(params, klist=self.ks_lin,
                                           zred=self.z)   
                                    
        return pk_lin_table
    
   