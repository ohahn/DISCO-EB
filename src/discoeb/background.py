import jax
import jax.numpy as jnp
import diffrax as drx
from jax_cosmo.scipy.integrate import romb

from .thermodynamics_recfast import evaluate_thermo as evaluate_thermo_recfast
from .thermodynamics_mb95 import compute_thermo as compute_thermo_mb95

from .spline_interpolation import spline_interpolation
from .cosmo import nu_background, dtauda_, get_aprimeoa

def setup_background_evolution( *, amin, amax, param ):
    c2ok = 1.62581581e4 # K / eV
    num_neutrino = 512  # number of neutrino history arrays
    
    param['amin'] = amin
    param['amax'] = amax   

    # mean densities
    Omegak = 0.0 #1.0 - Omegam - OmegaL
    param['grhom'] = 3.33795017e-11 * param['H0']**2    # 8πG rho_c / c^2 * in 1/Mpc^2
    param['grhog'] = 1.49594245e-13 * param['Tcmb']**4  # photon density in 1/Mpc^2
    param['grhor'] = 3.39739477e-14 * param['Tcmb']**4  # neutrino density per flavour in 1/Mpc^2
    param['adotrad'] = jnp.sqrt((param['grhog']+param['grhor']*(param['Neff']+param['Nmnu'])) / 3.0)
    # param['adotrad'] = 2.8948e-7 * param['Tcmb']**2 # Hubble during radiation domination

    param['amnu'] = param['mnu'] * c2ok / param['Tcmb'] # conversion factor for Neutrinos masses (m_nu*c**2/(k_B*T_nu0)

    # Compute the scale factor linearly spaced in log(a)
    a = jnp.geomspace(amin*0.9, amax*1.1, num_neutrino)
    loga = jnp.log(a)
    param['a'] = a

    # Compute the neutrino density and pressure
    rhonu_, pnu_, ppnu_ = jax.vmap( lambda a_ : nu_background( a_, param['amnu'] ), in_axes=0 )( a )

    param['logrhonu_of_loga_spline']     = spline_interpolation( loga, jnp.log(rhonu_) )
    param['logpnu_of_loga_spline']       = spline_interpolation( loga, jnp.log(pnu_) )
    param['logppseudonu_of_loga_spline'] = spline_interpolation( loga, jnp.log(ppnu_) )

    # compute the energy density today due to massive neutrinos
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0))
    Omegamnu = (param['grhor'] * rhonu) / param['grhom']
    param['Omegamnu'] = Omegamnu

    # ensure curvature is correct
    Omegar = (param['Neff']+param['Nmnu']*jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0))) * param['grhor'] / param['grhom']
    Omegag = param['grhog'] / param['grhom']
    param['OmegaDE'] = 1.0 - param['Omegak'] - Omegar - Omegag - param['Omegam']

    # Compute the conformal time interval
    param['taumin'] = amin / param['adotrad']
    param['taumax'] = (param['taumin'] + 
        romb( lambda a_: dtauda_(a_,param['grhom'], param['grhog'], param['grhor'], 
                        param['Omegam'], param['OmegaDE'], param['w_DE_0'], param['w_DE_a'],
                        param['Omegak'], param['Neff'], param['Nmnu'], 
                        param['logrhonu_of_loga_spline']), amin, amax )
    )

    return param


def evolve_background( *, param, thermo_module = 'RECFAST', rtol: float = 1e-5, atol: float = 1e-7, order: int = 5, class_thermo = None ):
    num_thermo   = 1024 # length of thermal history arrays
    c2ok = 1.62581581e4 # K / eV

    amin = 1e-9
    amax = 1.01

    if thermo_module == 'CLASS':
        amin = jnp.min( class_thermo['scale factor a'] )
        amax = jnp.max( class_thermo['scale factor a'] )
    
    param = setup_background_evolution( amin=amin, amax=amax, param=param )

    if thermo_module == 'RECFAST':
        # Compute the thermal history
        param, tau, aexp, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII, xeprime_recfast = evaluate_thermo_recfast( param=param, num_thermo=num_thermo )

        param['aexp'] = aexp
        param['tau'] = tau
        param['xe'] = xe
        param['xeHI'] = xeHI
        param['xeHeI'] = xeHeI
        param['xeHeII'] = xeHeII
        param['cs2'] = cs2
        param['Tm'] = Tm

        param['tau_of_a_spline']      = spline_interpolation( aexp, tau )
        param['a_of_tau_spline']      = spline_interpolation( tau, aexp )
        param['xe_of_tau_spline']     = spline_interpolation( tau, xe )
        param['cs2a_of_tau_spline']   = spline_interpolation( tau, aexp*cs2 )
        param['tempba_of_tau_spline'] = spline_interpolation( tau, aexp*Tm )

    elif thermo_module == 'MB95':

        # Compute the thermal history
        th, param = compute_thermo_mb95( param=param, nthermo=num_thermo )

        xe = th['xe']
        xeHI = th['xHII']
        xeHeI = th['xHeII']
        xeHeII = th['xHeIII']
        aexp = th['a']
        tau = th['tau']
        cs2 = th['cs2']
        Tm = th['tb']

        param['xe'] = xe
        param['xeHI'] = xeHI
        param['xeHeI'] = xeHeI
        param['xeHeII'] = xeHeII
        param['aexp'] = aexp
        param['tau'] = tau

        param['xe_of_tau_spline']     = spline_interpolation( tau, xe )
        param['cs2a_of_tau_spline']   = spline_interpolation( tau, aexp*cs2 )
        param['tempba_of_tau_spline'] = spline_interpolation( tau, aexp*Tm )
        param['tau_of_a_spline'] = spline_interpolation( aexp, tau )
        param['a_of_tau_spline'] = spline_interpolation( tau, aexp )

    elif thermo_module == 'CLASS':
        # use input CLASS thermodynamics
        # interpolating splines for the thermal history        
        param['cs2_of_tau_spline']   = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['c_b^2'][::-1] )
        param['tempb_of_tau_spline'] = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['Tb [K]'][::-1] )
        param['xe_of_tau_spline']    = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['x_e'][::-1] )
        param['a_of_tau_spline']     = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['scale factor a'][::-1] )
        param['tau_of_a_spline']     = spline_interpolation( class_thermo['scale factor a'][::-1],   class_thermo['conf. time [Mpc]'][::-1] )

        param['aexp'] = class_thermo['scale factor a'][::-1]
        param['tau'] = class_thermo['conf. time [Mpc]'][::-1]
    


    # compute optical depth and visibility functions
    akthom = 2.3038921003709498e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2
    # grho_v = jax.vmap( lambda a: grho( param, a ) )( a )(aexp)
    # aprimeoa = jnp.sqrt( grho_v / 3.0 )
    aprimeoa = get_aprimeoa( param=param, aexp=aexp )

    tau_pre_recomb = param['tau_of_a_spline'].evaluate( 1e-4 )
    xe_full   = 1 + param['YHe'] / (1 - param['YHe'])
    xe        = jnp.where( tau <= tau_pre_recomb, xe_full, param['xe_of_tau_spline'].evaluate( tau ) )
    xeprime   = jnp.where( tau <= tau_pre_recomb, 0.0, param['xe_of_tau_spline'].derivative( tau ) )

    # xe = param['xe_of_tau_spline'].evaluate( tau )
    # xeprime = param['xe_of_tau_spline'].derivative( tau )
    # xepprime  = param['xe_of_tau_spline'].derivative2( tau )
    opac      = xe * akthom / aexp**2

    opacspline = spline_interpolation( tau, opac, integrate_from_start=False)
    opacprime  = opacspline.derivative( tau )
    opacpprime = opacspline.derivative2( tau )
    # opacprime = xeprime * akthom / aexp**2 - 2 * xe * akthom / aexp**2 * aprimeoa

    optical_depth = opacspline.integral( tau )
    optical_depth_today = opacspline.integral( param['tau_of_a_spline'].evaluate( 1.0 ) )
    optical_depth -= optical_depth_today

    # optical_depth = jnp.array([optical_depth[0], *optical_depth])
    expmmu    = jnp.exp(-optical_depth)
    vis       = opac * expmmu
    dvis      = (opacprime + opac**2) * expmmu
    ddvis     = (opacpprime + 3 * opac * opacprime + opac**3) * expmmu

    param['optical_depth'] = optical_depth
    param['opac'] = opac
    param['gvis'] = vis
    param['gvisprime'] = dvis
    param['gvispprime'] = ddvis

    if thermo_module == 'RECFAST':
        param['xeprime_recf'] = xeprime_recfast

    param['xeprime'] = xeprime
    
    return param 
        

