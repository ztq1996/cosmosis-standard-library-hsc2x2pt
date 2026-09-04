"""Put Dark Emulator 2's linear and nonlinear matter power spectra into the
datablock as matter_power_lin / matter_power_nl, linearly extrapolated (in
log-log) beyond the range the emulator covers.

Ini options, and the two that bite:

  * npart / lbox go through get_double, and cosmosis distinguishes int from
    double strictly -- write "3000.0", not "3000", or setup() raises
    BlockWrongValueType.
  * extrap_kmin defaults to 1e10, which is not a typo: linear_extend() only
    extends when kmin < k.min(), so this default means "do not extrapolate on
    the low-k side". Set it (e.g. 1e-5) to switch low-k extrapolation on.

Sizing nz: the z grid is not just an interpolation grid. magnification_phys
feeds it straight into integrate.simpson() as the Limber integration sample
points, so nz sets that quadrature step. It does not need to be fine -- the
integrand is smooth in z -- but check convergence when changing it. Cost is
linear in nz: nz = 150 takes ~7 s per evaluation, nz = 20 takes ~1.5 s.

Caveat on the extrapolated region: the log-log slope is fit per redshift, so
above the emulator's kmax the z ordering of Pnl/Plin can invert (z = 3 crossing
above z = 0), which is unphysical. Keep extrap_kmax to what the consumer
actually needs rather than padding it.

save_sigma8 (default T) controls whether sigma_8 / S_8 are written. Set it to F
in a pipeline whose reported sigma_8 comes from Dark Emulator I: the two
emulators build the linear P(k) differently and disagree by 0.27%, so leaving it
on makes the consistency check below fire (or, if this module runs first,
silently hands over a value that darkemu_sigma8 then overwrites).

Author: Ryo Terasawa / Sunao Sugiyama / Keitaro Ishikawa
"""
import warnings
import dq2emu
import numpy as np
from numpy import log, exp
from cosmosis.datablock import names, option_section as opt

warnings.filterwarnings("ignore", category=UserWarning)
cosmo_pars = names.cosmological_parameters

# k range Dark Emulator 2 itself covers [h/Mpc]; outside this we extrapolate.
K_NATIVE_MAX = 1e2

def setup(options):
    zmax = options.get_double(opt, "zmax", default=3.00)
    # 20 is enough for the magnification-bias use case (see the note above);
    # raise it if a consumer needs a finer z quadrature.
    nz = options.get_int(opt, "nz", default=20)
    # power spectrum
    nk = options.get_int(opt, "nk", default=400)
    logkhmin = np.log10(options.get_double(opt, "kmin", default=1e-3))
    logkhmax = np.log10(options.get_double(opt, "kmax", default=1e2))

    # NOTE: get_double -> these must be written as "3000.0" / "1000.0" in the ini.
    npart = options.get_double(opt, "npart", default=3000.0)
    lbox = options.get_double(opt, "lbox", default=1000.0)

    extrap = {}
    extrap["kmax"] = options.get_double(opt, "extrap_kmax")
    # default 1e10 == "no low-k extrapolation" (see module docstring).
    extrap["kmin"] = options.get_double(opt, "extrap_kmin", default=1e10)
    extrap["nmin"] = options.get_int(opt, "extrap_nmin", default=50)
    extrap["npoint"] = options.get_int(opt, "extrap_npoint", default=3)
    extrap["nmax"] = options.get_int(opt, "extrap_nmax", default=200)
    extrap["sn"] = options.get_double(opt, "extrap_sn", default=10)
    extrap["shotnoise"] = (lbox/npart)**3

    # F when the pipeline reports Dark Emulator I's sigma_8 (see the docstring).
    save_sigma8 = options.get_bool(opt, "save_sigma8", default=True)


    cc = dq2emu.PsConfig(
            nz=nz, zmax=zmax,
            logkh_min=logkhmin,
            logkh_max=logkhmax,
            nk=nk, npart=npart,
            lbox=lbox,
            )
    emulator = dq2emu.DQ2Emu(cc)
            
    return emulator, extrap, save_sigma8

def get_input_pars(block):
    params = {
        'Omega_m': block[cosmo_pars, "omega_m"],
        'omega_b': block[cosmo_pars, "ombh2"],
        "Mnu": block.get_double(cosmo_pars, "mnu", 0.06),
        'Omega_k': block[cosmo_pars, "omega_k"],
        'ns': block[cosmo_pars, "n_s"],        
        "w0": block.get_double(cosmo_pars, "w", -1.0),
        "wa": block.get_double(cosmo_pars, "wa", 0.0),
        'h0': block[cosmo_pars, "h0"],
        'As': block[cosmo_pars, "A_s"], 
    }
    return params

def linear_extend(x, y, xmin, xmax, nmin, nmax, nfit, x2=None):
    if xmin < x.min():
        xf = x[:nfit]
        yf = y[:nfit]
        p = np.polyfit(xf, yf, 1)
        xnew = np.linspace(xmin, x.min(), nmin, endpoint=False)
        ynew = np.polyval(p, xnew)
        x = np.concatenate((xnew, x))
        y = np.concatenate((ynew, y))
    if xmax > x.max():
        xf = x[-nfit:]
        yf = y[-nfit:]
        p = np.polyfit(xf, yf, 1)
        if x2 is not None:            
            _xnew = np.linspace(x2.max(), xmax, nmax, endpoint=True)
            # skip the first point as it is just the x2max
            _xnew = _xnew[1:]
            xnew = np.concatenate((x2, _xnew)) 
        else:
            xnew = np.linspace(x.max(), xmax, nmax, endpoint=True)
            # skip the first point as it is just the xmax
            xnew = xnew[1:]
        ynew = np.polyval(p, xnew)
        x = np.concatenate((x, xnew))
        y = np.concatenate((y, ynew))
    return x, y

def execute(block, config):
    # preparation
    emulator, extrap, save_sigma8 = config
    nz = emulator.nz
    k_h = emulator.ks

    params = get_input_pars(block)
    
    # setup extrapolation
    kmin = extrap['kmin']
    kmax = extrap['kmax']
    nmin = extrap['nmin']
    nmax = extrap['nmax']
    npoint = extrap['npoint']
    sn = extrap['sn']
    shotnoise = extrap['shotnoise']
    
    # save sigma8 and S8
    if save_sigma8:
        sigma8 = emulator.get_sigma8(params)
        # Fires when another module already wrote sigma_8 from a different
        # linear P(k) -- in particular darkemu_sigma8 (Dark Emulator I), which
        # disagrees by 0.27%. Set save_sigma8 = F to keep that module's value.
        if block.has_value(cosmo_pars, "sigma_8"):
            other = block[cosmo_pars, "sigma_8"]
            if not np.isclose(sigma8, other):
                raise ValueError(
                    "sigma_8 is already in the datablock from another module "
                    "(%.6f) and disagrees with Dark Emulator 2 (%.6f). Set "
                    "save_sigma8 = F in this module's ini section to keep the "
                    "existing value." % (other, sigma8))
        block[cosmo_pars, "sigma_8"] = sigma8
        block[cosmo_pars, "S_8"] = sigma8 * np.sqrt(block[cosmo_pars, "omega_m"] / 0.3)


    # linear and nonlinear power spectrum
    pk_lin_table, pk_nl_table = emulator.compute_pk_table(params)
    
    # extrapolate
    P_lin_out = []
    P_nl_out = []
    
    
    for i in range(nz):
        Plin_i = pk_lin_table[i]

        # Drop the shot-noise-dominated tail and replace it by the log-log
        # linear extension fitted below (K_NATIVE_MAX = emulator kmax).
        indx1 = (pk_nl_table[i] <= sn * shotnoise) & (k_h <= K_NATIVE_MAX) & (k_h > 1e-2)
        indx11 = np.where(indx1)[0]
        if len(indx11) == 0:
            logk2 = None
            k1 = k_h[k_h <= K_NATIVE_MAX]
            Pnl_i = pk_nl_table[i][k_h <= K_NATIVE_MAX]
        else:
            indx_first_subsn = indx11[0]
            k1 = k_h[:indx_first_subsn]
            Pnl_i = pk_nl_table[i][:indx_first_subsn]
            k2 = k_h[indx_first_subsn:]
            k2 = k2[k2 <= K_NATIVE_MAX]
            logk2 = log(k2)
        
        logk_lin, logplin = linear_extend(log(k_h), log(Plin_i), log(
            kmin), log(kmax), nmin, nmax, npoint)
        P_lin_out.append(exp(logplin))
        
        logk, logpnl = linear_extend(log(k1), log(Pnl_i), log(
            kmin), log(kmax), nmin, nmax, npoint, logk2)
        P_nl_out.append(exp(logpnl))

    k_lin = exp(logk_lin)
    k = exp(logk)
    
    block.put_grid(
            "matter_power_lin", "z", emulator.z, "k_h", k_lin,
            "p_k", np.array(P_lin_out),
            )
    
    block.put_grid(
            "matter_power_nl", "z", emulator.z, "k_h", k,
            "p_k", np.array(P_nl_out),
            )
    
    return 0


def cleanup(config):
    pass
