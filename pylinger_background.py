from pylinger_thermodynamics_recfast import compute_thermo as compute_thermo_recfast, evaluate_thermo as evaluate_thermo_recfast
from pylinger_thermodynamics_mb95 import compute_thermo as compute_thermo_mb95
from pylinger_cosmo import nu_background
import jax
import jax.numpy as jnp
import diffrax as drx
from jax_cosmo.scipy.integrate import romb
from functools import partial




def a_of_tau( tau_ : float | jax.Array, param : dict ) -> float | jax.Array:
    a, _, _, _ = param['sol'].evaluate(tau_)
    return a

def dtauda_(a, grhom, grhog, grhor, Omegam, OmegaDE, w_DE_0, w_DE_a, Omegak, Neff, Nmnu, logrhonu_spline):
    """Derivative of conformal time with respect to scale factor"""
    rho_DE = a**(-3*(1+w_DE_0+w_DE_a)) * jnp.exp(3*(a-1)*w_DE_a)
    grho2 = grhom * Omegam * a \
        + (grhog + grhor*(Neff+Nmnu*jnp.exp(logrhonu_spline.evaluate(jnp.log(a))))) \
        + grhom * OmegaDE * rho_DE * a**4 \
        + grhom * Omegak * a**2
    return jnp.sqrt(3.0 / grho2)


@partial(jax.jit, static_argnames=('thermo_module',))
def evolve_background( *, param, thermo_module = 'RECFAST', rtol: float = 1e-5, atol: float = 1e-7, order: int = 5, class_thermo = None ):
    c2ok = 1.62581581e4 # K / eV
    amin = 1e-9
    amax = 1.01
    num_thermo   = 2048 # length of thermal history arrays
    num_neutrino = 512  # number of neutrino history arrays
    
    param['amin'] = amin
    param['amax'] = amax   

    # mean densities
    Omegak = 0.0 #1.0 - Omegam - OmegaL
    param['grhom'] = 3.33795017e-11 * param['H0']**2    # critical density at z=0 in h^2/Mpc^3
    param['grhog'] = 1.49594245e-13 * param['Tcmb']**4  # photon density in h^2/Mpc^3
    param['grhor'] = 3.39739477e-14 * param['Tcmb']**4  # neutrino density per flavour in h^2/Mpc^3
    param['adotrad'] = jnp.sqrt((param['grhog']+param['grhor']*(param['Neff']+param['Nmnu'])) / 3.0)
    # param['adotrad'] = 2.8948e-7 * param['Tcmb']**2 # Hubble during radiation domination

    param['amnu'] = param['mnu'] * c2ok / param['Tcmb'] # conversion factor for Neutrinos masses (m_nu*c**2/(k_B*T_nu0)


    if thermo_module == 'CLASS':
        amin = jnp.min( class_thermo['scale factor a'] )
        amax = jnp.max( class_thermo['scale factor a'] )

    
    # Compute the scale factor linearly spaced in log(a)
    a = jnp.geomspace(amin, amax, num_neutrino)
    loga = jnp.log(a)
    param['a'] = a

    # Compute the neutrino density and pressure
    rhonu_, pnu_, ppnu_ = jax.vmap( lambda a_ : nu_background( a_, param['amnu'] ), in_axes=0 )( a )

    rhonu_coeff = drx.backward_hermite_coefficients(ts=loga, ys=jnp.log(rhonu_))
    pnu_coeff = drx.backward_hermite_coefficients(ts=loga, ys=jnp.log(pnu_))
    ppnu_coeff = drx.backward_hermite_coefficients(ts=loga, ys=jnp.log(ppnu_))
    rhonu_spline = drx.CubicInterpolation( ts=loga, coeffs=rhonu_coeff )
    
    param['logrhonu_of_loga_spline']     = rhonu_spline
    param['logpnu_of_loga_spline']       = drx.CubicInterpolation( ts=loga, coeffs=pnu_coeff )
    param['logppseudonu_of_loga_spline'] = drx.CubicInterpolation( ts=loga, coeffs=ppnu_coeff )

    # compute the energy density today due to massive neutrinos
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0))
    Omegamnu = (param['grhor'] * rhonu) / param['grhom']
    param['Omegamnu'] = Omegamnu

    # ensure curvature is correct
    Omegar = (param['Neff']+param['Nmnu']*jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0))) * param['grhor'] / param['grhom']
    param['OmegaDE'] = 1.0 - param['Omegak'] - Omegar - param['Omegam']
    print('OmegaDE = ',param['OmegaDE'])

    # Compute the conformal time interval
    param['taumin'] = amin / param['adotrad']
    param['taumax'] = param['taumin'] + romb( lambda a_: dtauda_(a_,param['grhom'], param['grhog'], param['grhor'], 
                                                                 param['Omegam'], param['OmegaDE'], param['w_DE_0'], param['w_DE_a'],
                                                                 param['Omegak'], param['Neff'], param['Nmnu'], 
                                                                 param['logrhonu_of_loga_spline']), amin, amax )

    
    if thermo_module == 'RECFAST':
        # Compute the thermal history
        sol, param = compute_thermo_recfast( param=param )

        # # param['a_of_tau_spline'] = lambda tau : jnp.exp(sol.evaluate( tau )[0])
        param['sol'] = sol
        # # param['a_of_tau_spline']     = lambda tau : jnp.exp(param['sol'].evaluate( tau )[0])

        # tau  = jnp.geomspace(param['taumin'], param['taumax'], num_thermo)
        # aexp = jax.vmap( lambda tau_: jnp.exp((param['sol'].evaluate(tau_))[0]), in_axes=0 )( tau )
        tau  = param['sol'].ts
        aexp = jnp.exp(param['sol'].ys[:,0])

        param['tau_coeff'] = drx.backward_hermite_coefficients(ts=aexp, ys=tau)
        param['tau_of_a_spline'] = drx.CubicInterpolation( ts=aexp, coeffs=param['tau_coeff'] )

        param['a_coeff'] = drx.backward_hermite_coefficients(ts=tau, ys=aexp)
        param['a_of_tau_spline'] = drx.CubicInterpolation( ts=tau, coeffs=param['a_coeff'] )

        param['a'] = aexp
        param['tau'] = tau

        tau, a, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII = evaluate_thermo_recfast( param=param, num_thermo=num_thermo )

        param['xe'] = xe
        param['xeHI'] = xeHI
        param['xeHeI'] = xeHeI
        param['xeHeII'] = xeHeII
        

        param['xe_coeff']  = drx.backward_hermite_coefficients(ts=tau, ys=xe)
        param['xe_of_tau_spline']    = drx.CubicInterpolation( ts=tau, coeffs=param['xe_coeff'] )

        param['cs2a_coeff'] = drx.backward_hermite_coefficients(ts=tau, ys=(a*cs2))
        param['cs2a_of_tau_spline']   = drx.CubicInterpolation( ts=tau, coeffs=param['cs2a_coeff'] )

        param['tba_coeff']  = drx.backward_hermite_coefficients(ts=tau, ys=(a*Tm))
        param['tempba_of_tau_spline'] = drx.CubicInterpolation( ts=tau, coeffs=param['tba_coeff'] )


    elif thermo_module == 'MB95':

        # Compute the thermal history
        th, param = compute_thermo_mb95( param=param, nthermo=num_thermo )

        xe = th['xe']
        xeHI = th['xHII']
        xeHeI = th['xHeII']
        xeHeII = th['xHeIII']
        a = th['a']
        tau = th['tau']
        cs2 = th['cs2']
        Tm = th['tb']

        # tau, a, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII = compute_thermo_mb95( param=param, num_thermo=num_thermo )

        param['xe'] = xe
        param['xeHI'] = xeHI
        param['xeHeI'] = xeHeI
        param['xeHeII'] = xeHeII

        param['xe_coeff']  = drx.backward_hermite_coefficients(ts=tau, ys=xe)
        param['xe_of_tau_spline']    = drx.CubicInterpolation( ts=tau, coeffs=param['xe_coeff'] )

        param['cs2a_coeff'] = drx.backward_hermite_coefficients(ts=tau, ys=(a*cs2))
        param['cs2a_of_tau_spline']   = drx.CubicInterpolation( ts=tau, coeffs=param['cs2a_coeff'] )

        param['tba_coeff']  = drx.backward_hermite_coefficients(ts=tau, ys=(a*Tm))
        param['tempba_of_tau_spline'] = drx.CubicInterpolation( ts=tau, coeffs=param['tba_coeff'] )

        param['a'] = a
        param['tau'] = tau

        param['tau_coeff'] = drx.backward_hermite_coefficients(ts=a, ys=tau)
        param['tau_of_a_spline'] = drx.CubicInterpolation( ts=a, coeffs=param['tau_coeff'] )

        param['a_coeff'] = drx.backward_hermite_coefficients(ts=tau, ys=a)
        param['a_of_tau_spline'] = drx.CubicInterpolation( ts=tau, coeffs=param['a_coeff'] )

    elif thermo_module == 'CLASS':
        # use input CLASS thermodynamics
        # interpolating splines for the thermal history
        param['cs2_coeff'] = drx.backward_hermite_coefficients(ts=class_thermo['conf. time [Mpc]'][::-1], ys=class_thermo['c_b^2'][::-1])
        param['tb_coeff']  = drx.backward_hermite_coefficients(ts=class_thermo['conf. time [Mpc]'][::-1], ys=class_thermo['Tb [K]'][::-1])
        param['xe_coeff']  = drx.backward_hermite_coefficients(ts=class_thermo['conf. time [Mpc]'][::-1], ys=class_thermo['x_e'][::-1])
        param['a_coeff']   = drx.backward_hermite_coefficients(ts=class_thermo['conf. time [Mpc]'][::-1], ys=class_thermo['scale factor a'][::-1])
        param['tau_coeff'] = drx.backward_hermite_coefficients(ts=class_thermo['scale factor a'][::-1],   ys=class_thermo['conf. time [Mpc]'][::-1])
        
        param['cs2_of_tau_spline']   = drx.CubicInterpolation( ts=class_thermo['conf. time [Mpc]'][::-1], coeffs=param['cs2_coeff'] )
        param['tempb_of_tau_spline'] = drx.CubicInterpolation( ts=class_thermo['conf. time [Mpc]'][::-1], coeffs=param['tb_coeff'] )
        param['xe_of_tau_spline']    = drx.CubicInterpolation( ts=class_thermo['conf. time [Mpc]'][::-1], coeffs=param['xe_coeff'] )
        param['a_of_tau_spline']     = drx.CubicInterpolation( ts=class_thermo['conf. time [Mpc]'][::-1], coeffs=param['a_coeff'] )
        param['tau_of_a_spline']     = drx.CubicInterpolation( ts=class_thermo['scale factor a'][::-1],   coeffs=param['tau_coeff'] )

        param['a'] = class_thermo['scale factor a'][::-1]
        param['tau'] = class_thermo['conf. time [Mpc]'][::-1]
    

    
    return param 
        

