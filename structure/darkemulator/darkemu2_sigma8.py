"""
Write sigma_8 and S_8 into the datablock using Dark Emulator 2's linear power
spectrum.  Dark Emulator 2 version of darkemu_sigma8.py -- use this one in
pipelines built on dq2_interface_extrapolate.py, and keep darkemu_sigma8.py for
the Dark Emulator I pipelines.

Why a separate module: DE1 and DE2 take the cosmology in different forms (DE1
wants the 6-vector (ombh2, omch2, Omega_lambda, log1e10As, n_s, w) and fixes
omnuh2 = 0.00064 internally; DE2 wants a 9-parameter dict with a free Mnu, w0,
wa).  Mixing them in one pipeline means sigma_8 is written twice from two
different linear P(k), so pick one.

The cosmology is read through get_input_pars() imported from
dq2_interface_extrapolate, so this module and the mPk module are guaranteed to
describe the same universe.  It only needs `consistency` in front of it (for
omega_m / h0 / A_s) and works without the mPk module present.

Cost: ~0.2 s to load the emulator in setup(), ~0.05 s per evaluation -- cheap
enough to avoid running camb just to report these two derived parameters.

Author: Keitaro Ishikawa, 2026-07-30
"""
import numpy as np
from cosmosis.datablock import names, option_section

import dq2emu
# Same reader as the mPk interface -> identical cosmology by construction.
from dq2_interface_extrapolate import get_input_pars


def setup(options):
    # NOTE: cosmosis distinguishes int and double strictly. npart/lbox go
    # through get_double, so write them as "3000.0" / "1000.0" in the ini
    # (a bare "3000" raises BlockWrongValueType).
    npart = options.get_double(option_section, "npart", default=3000.0)
    lbox = options.get_double(option_section, "lbox", default=1000.0)

    # sigma_8 comes from the linear emulator alone, so the z/k grids in
    # PsConfig are never touched here -- the defaults are fine and cost
    # nothing but two np.logspace calls. npart/lbox are kept configurable so
    # that this module and the mPk module can be pointed at the same emulator.
    emu = dq2emu.DQ2Emu(dq2emu.PsConfig(npart=npart, lbox=lbox))
    return emu


def execute(block, config):
    emu = config
    pars = names.cosmological_parameters

    sigma_8 = emu.get_sigma8(get_input_pars(block))

    block[pars, 'sigma_8'] = sigma_8
    block[pars, 's_8'] = sigma_8 * np.sqrt(block[pars, 'omega_m'] / 0.3)

    return 0


def cleanup(config):
    return 0
