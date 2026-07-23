"""
Write sigma_8 and S_8 into the datablock using Dark Emulator's own linear power
spectrum, so that a pipeline running darkemu_x_hod does not need CAMB just to
report those two derived parameters.

darkemu_x_hod itself never touches matter_power_*, so once magnification_phys is
dropped the only remaining reason to run camb + pyhalofit is the
`extra_output = cosmological_parameters/sigma_8 cosmological_parameters/s_8`
line. That costs ~0.75 s per likelihood evaluation; this module costs ~0.01 s.

sigma_8 is computed by base_class.get_sigma8(), i.e. the tophat-R8 variance of
the emulated linear P(k) at z = 0, and S_8 = sigma_8 * sqrt(Omega_m / 0.3).

The cosmology is read exactly the way darkemu_x_hod reads it, so the two modules
are guaranteed to describe the same universe. Place this module anywhere after
`consistency` (which supplies omega_lambda); it does not depend on the theory
module and can equally be used with a pipeline that has no darkemu in it.

Author: Keitaro Ishikawa, 2026-07-22
"""
import numpy as np
from cosmosis.datablock import names, option_section
from dark_emulator import darkemu


def setup(options):
    # Loading the emulator takes ~1 s, so do it once here rather than per sample.
    emu = darkemu.base_class()
    return emu


def execute(block, config):
    emu = config
    pars = names.cosmological_parameters

    # Same order and same names as darkemu_x_hod.py; Dark Emulator I fixes
    # omnuh2 = 0.00064 internally and derives h from these six numbers.
    params = np.array([block[pars, 'ombh2'],
                       block[pars, 'omch2'],
                       block[pars, 'omega_lambda'],
                       block[pars, 'log1e10As'],
                       block[pars, 'n_s'],
                       block[pars, 'w']])
    emu.set_cosmology(params.reshape((1, 6)))

    sigma_8 = emu.get_sigma8()
    omega_m = block[pars, 'omega_m']

    block[pars, 'sigma_8'] = sigma_8
    block[pars, 's_8'] = sigma_8 * np.sqrt(omega_m / 0.3)

    return 0


def cleanup(config):
    return 0
