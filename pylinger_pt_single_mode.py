import jax
import jax.numpy as jnp
import jax_cosmo.scipy.interpolate as jaxinterp
from pylinger_cosmo import cosmo, nu_perturb_jax
from functools import partial
import diffrax
# from scipy.integrate import solve_ivp
# import equinox as eqx  # https://github.com/patrick-kidger/equinox

@partial(jax.jit, static_argnames=('lmax', 'lmaxnu', 'nqmax'))
def model_synchronous(*, tau, yin, param, kmode, lmax, lmaxnu, nqmax, ):
    iq0 = 11 + 3 * lmax
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    y = jnp.copy(yin)
    f = jnp.zeros_like(y)

    # print(y[0,0])

    # ... metric
    a = y[0]
    ahdot = y[1]
    eta = y[2]

    # ... cdm
    deltac = y[3]
    thetac = y[4]

    # ... baryons
    deltab = y[5]
    thetab = y[6]

    # ... photons
    deltag = y[7]
    thetag = y[8]
    shearg = y[9] / 2.0

    # ... polarization term
    polter = y[9] + y[8 + lmax] + y[10 + lmax]

    # ... massless neutrinos
    deltar = y[9 + 2 * lmax]
    thetar = y[10 + 2 * lmax]
    shearr = y[11 + 2 * lmax] / 2.0

    # tempb, cs2, xe = params.th.evaluate_at_tau(tau)
    tempb = param['tempb_of_tau_spline']( tau )
    cs2 = param['cs2_of_tau_spline']( tau )
    xe = param['xe_of_tau_spline']( tau )

    # ... Thomson opacity coefficient
    akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    # ... Thomson opacity
    opac = xe * akthom / a**2

    # ... Photon mass density over baryon mass density
    photbar = param['grhog'] / (param['grhom'] * param['Omegab'] * a)
    pb43 = 4.0 / 3.0 * photbar

    # ... compute expansion rate
    rhonu = param['rhonu_of_a_spline']( a )
    pnu = param['pnu_of_a_spline']( a ) 
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaL'] * a**2
        + param['grhom'] * param['Omegak']
    )
    adotoa = jnp.sqrt(grho / 3.0)

    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 - param['grhom'] * param['OmegaL'] * a**2

    f = f.at[0].set( adotoa * a )

    
    # ... evaluate metric perturbations
    drhonu, dpnu, fnu, shearnu = nu_perturb_jax( a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3] )

    dgrho = (
        param['grhom'] * (param['Omegac'] * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
    )
    dgpres = (
        param['grhog'] * deltag + param['grhor'] * param['Neff'] * deltar
    ) / a**2 / 3.0 + param['grhor'] * param['Nmnu'] * dpnu / a**2

    dahdotdtau = -(dgrho + 3.0 * dgpres) * a
    
    f = f.at[1].set( dahdotdtau )

    # ... force energy conservation
    hdot = (2.0 * kmode**2 * eta + dgrho) / adotoa

    dgtheta = (
        param['grhom'] * (param['Omegac'] * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag + param['Neff'] * param['grhor'] * thetar) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
    )
    etadot = 0.5 * dgtheta / kmode**2
    f = f.at[2].set( etadot )

    dgshear = (
        4.0 / 3.0 * (param['grhog'] * shearg + param['Neff'] * param['grhor'] * shearr) / a**2
        + param['Nmnu'] * param['grhor'] * shearnu / a**2
    )

    # ... cdm equations of motion
    deltacdot = -thetac - 0.5 * hdot
    f = f.at[3].set( deltacdot )
    thetacdot = -adotoa * thetac
    f = f.at[4].set( thetacdot )

    # ... baryon equations of motion
    deltabdot = -thetab - 0.5 * hdot
    f = f.at[5].set( deltabdot )
    # ... need photon perturbation for first-order correction to tightly-coupled baryon-photon approx.
    deltagdot = 4.0 / 3.0 * (-thetag - 0.5 * hdot)
    drag = opac * (thetag - thetab)

    def calc_thetabdot_lowt():
        # ... treat baryons and photons as uncoupled
        return -adotoa * thetab + kmode**2 * cs2 * deltab + pb43 * drag
    
    def calc_thetabdot_hight():
        # ... treat baryons and photons as tighly coupled.
        # ... zeroth order approx to baryon velocity
        thetabdot_hight = (
            -adotoa * thetab + kmode**2 * cs2 * deltab + kmode**2 * pb43 * (0.25 * deltag - shearg)
        ) / (1.0 + pb43)
        adotdota = 0.5 * (adotoa * adotoa - gpres)

        # ... first-order approximation to baryon-photon slip, thetabdot-thetagdot.
        slip = 2.0 * pb43 / (1.0 + pb43) * adotoa * (thetab - thetag) + 1.0 / opac * (
            -adotdota * thetab
            - adotoa * kmode**2 * 0.5 * deltag
            + kmode**2 * (cs2 * deltabdot - 0.25 * deltagdot)
        ) / (1.0 + pb43)
        # ... first oder approximation to baryon velocity
        thetabdot_hight += pb43 / (1.0 + pb43) * slip
        return thetabdot_hight

    thetabdot = jax.lax.cond(tempb<1.0e4, calc_thetabdot_lowt, calc_thetabdot_hight)

    f = f.at[6].set( thetabdot )

    # ... photon total intensity and polarization equations of motion
    f = f.at[7].set( deltagdot )
    thetagdot = (-thetabdot - adotoa * thetab + kmode**2 * cs2 * deltab) / pb43 + kmode**2 * ( 0.25 * deltag - shearg )
    f = f.at[8].set( thetagdot )

    def update_photons_uncoupled( f ):
        # ... treat baryons and photons as uncoupled
        f = f.at[9].set(
            8.0 / 15.0 * thetag
            - 0.6 * kmode * y[10]
            - opac * y[9]
            + 4.0 / 15.0 * hdot
            + 8.0 / 5.0 * etadot
            + 0.1 * opac * polter
        )

        # ... polarization equations for l=1,2,3...
        f = f.at[8 + lmax].set( -kmode * y[9 + lmax] - opac * y[8 + lmax] + 0.5 * opac * polter )
        f = f.at[9 + lmax].set( kmode / 3.0 * (y[8 + lmax] - 2.0 * y[10 + lmax]) - opac * y[9 + lmax] )
        f = f.at[10 + lmax].set( kmode * (0.4 * y[9 + lmax] - 0.6 * y[11 + lmax]) - opac * y[10 + lmax] + 0.1 * opac * polter)
        
        ell  = jnp.arange(2, lmax - 1)
        f = f.at[8+ell].set( kmode  / (2 * ell + 2) * ((ell + 1) * y[7 + ell] - (ell + 2) * y[9 + ell]) - opac * y[8 + ell] )
        f = f.at[9+lmax+ell].set( kmode  / (2 * ell + 2) * ((ell + 1) * y[8 + lmax + ell] - (ell + 2) * y[10 + lmax + ell]) - opac * y[9 + lmax + ell] )

        return f

    def update_photons_coupled( f ):
        # do nothing since array is already initialized to zero
        return f

    f = jax.lax.cond( tempb < 1e4, update_photons_uncoupled, update_photons_coupled, f)

    # ... truncate moment expansion
    f = f.at[7 + lmax].set( kmode * y[6 + lmax] - (lmax + 1) / tau * y[7 + lmax] - opac * y[7 + lmax] )
    f = f.at[8 + 2 * lmax].set( kmode * y[7 + 2 * lmax] - (lmax + 1) / tau * y[8 + 2 * lmax] - opac * y[8 + 2 * lmax] )

    # ... Massless neutrino equations of motion
    deltardot = 4.0 / 3.0 * (-thetar - 0.5 * hdot)
    f = f.at[9 + 2 * lmax].set( deltardot )
    thetardot = kmode**2 * (0.25 * deltar - shearr)
    f = f.at[10 + 2 * lmax].set( thetardot )
    f = f.at[11 + 2 * lmax].set( 8.0 / 15.0 * thetar - 0.6 * kmode * y[12 + 2 * lmax] + 4.0 / 15.0 * hdot + 8.0 / 5.0 * etadot )
    ell = jnp.arange(2, lmax - 1)
    f = f.at[10 + 2 * lmax + ell].set( kmode / (2 * ell + 2) * ((ell + 1) * y[9 + 2 * lmax + ell] - (ell + 2) * y[11 + 2 * lmax + ell]) )
    
    # ... truncate moment expansion
    f = f.at[9 + 3 * lmax].set( kmode * y[8 + 3 * lmax] - (lmax + 1) / tau * y[9 + 3 * lmax] )

    # ... Massive neutrino equations of motion
    q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    dlfdlq = -q / (1.0 + jnp.exp(-q))  # derivative of the Fermi-Dirac distribution

    f = f.at[iq0 : iq1].set(
        -kmode * v * y[iq1 : iq2] + hdot* dlfdlq / 6.0
    )
    f = f.at[iq1 : iq2].set(
        kmode * v * (y[iq0 : iq1] - 2.0 * y[iq2 : iq3]) / 3.0
    )
    f = f.at[iq2 : iq3].set(
        kmode * v * (2 * y[iq1 : iq2] - 3 * y[iq3 : iq4]) / 5.0 - (hdot / 15 + 2 / 5 * etadot) * dlfdlq
    )

    for l in range(3, lmaxnu - 1):
        f = f.at[iq0 + l * nqmax : iq0 + (l + 1) * nqmax].set(
            kmode * v / (2 * l + 1) * (
                l * y[iq0 + (l - 1) * nqmax : iq0 + (l) * nqmax]
                - (l + 1) * y[iq0 + (l + 1) * nqmax : iq0 + (l + 2) * nqmax]
            )
        )

    # Truncate moment expansion.
    f = f.at[-nqmax :].set(
        kmode * v * y[-2 * nqmax : -nqmax] - (lmaxnu + 1) / tau * y[-nqmax :]
    )

    return f.flatten()

# @partial(jax.jit, static_argnames=('num_k', 'nvar', 'lmax', 'nqmax'))
def adiabatic_ics( *, tau: float, param, kmodes, num_k, nvar, lmax, nqmax):
    """Initial conditions for adiabatic perturbations"""
    iq0 = 11 + 3 * lmax
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    y = jnp.zeros((num_k, nvar))
    a = tau * param['adotrad']
    a2 = a**2

    rhonu = param['rhonu_of_a_spline'](a)
    pnu = param['pnu_of_a_spline'](a)
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaL'] * a**2
        + param['grhom'] * param['Omegak']
    )
    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 - param['grhom'] * param['OmegaL'] * a**2
    adotoa = jnp.sqrt(grho / 3.0)

    s = grho + gpres

    fracnu = param['grhor'] * (param['Neff'] + param['Nmnu']) * 4.0 / 3.0 / a2 / s

    # ... use yrad=rho_matter/rho_rad to correct initial conditions for matter+radiation
    yrad = (
        param['grhom'] * param['Omegam'] * a
        / (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu))
    )

    # .. isentropic ("adiabatic") initial conditions
    psi = -1.0
    C = (15.0 + 4.0 * fracnu) / 20.0 * psi
    akt2 = kmodes * tau
    akt2 *= akt2
    h = C * akt2 * (1.0 - 0.2 * yrad)
    eta = 2.0 * C - (5.0 + 4.0 * fracnu) / 6.0 / (15.0 + 4.0 * fracnu) * C * akt2 * (1.0 - yrad / 3.0)
    f1 = (23.0 + 4.0 * fracnu) / (15.0 + 4.0 * fracnu)

    deltac = -0.5 * h
    deltag = -2.0 / 3.0 * h * (1.0 - akt2 / 36.0)
    deltab = 0.75 * deltag
    deltar = -2.0 / 3.0 * h * (1.0 - akt2 / 36.0 * f1)
    deltan = deltar
    thetac = 0.0
    thetag = -C / 18.0 * akt2 * akt2 / tau
    thetab = thetag
    thetar = f1 * thetag
    thetan = thetar
    shearr = 4.0 / 15.0 * kmodes**2 / s * psi * (1.0 + 7.0 / 36.0 * yrad)
    shearn = shearr
    ahdot = 2.0 * C * kmodes**2 * tau * a * (1.0 - 0.3 * yrad)

    # ... metric
    y = y.at[:, 0].set( a )
    y = y.at[:, 1].set( ahdot )
    y = y.at[:, 2].set( eta )

    # .. CDM
    y = y.at[:, 3].set( deltac )
    y = y.at[:, 4].set( thetac )

    # .. baryons
    y = y.at[:, 5].set( deltab )
    y = y.at[:, 6].set( thetab )

    # ... Photons (total intensity and polarization)
    y = y.at[:, 7].set( deltag )
    y = y.at[:, 8].set( thetag )
    y = y.at[:, 8 + lmax].set( 0.0 ) # shearg
    y = y.at[:, 9 + lmax].set( 0.0 ) # polarization term

    for l in range(1, lmax):
        y = y.at[:, 8 + l].set( 0.0 )
        y = y.at[:, 9 + lmax + l].set( 0.0 )

    # ... massless neutrinos
    y = y.at[:, 9 + 2 * lmax].set( deltar )
    y = y.at[:, 10 + 2 * lmax].set( thetar )
    y = y.at[:, 11 + 2 * lmax].set( shearr * 2.0 )

    # for l in range(2, lmax):
    y = y.at[:, 10 + 2 * (lmax + 1) :].set( 0.0 )

    # ... massive neutrinos
    # if params.cp.Nmnu > 0:
    q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    akv = jnp.outer(kmodes, v)
    dlfdlq = -q / (1.0 + jnp.exp(-q))
    y = y.at[:, iq0:iq1].set( -0.25 * dlfdlq[None, :] * deltan[:, None] )
    y = y.at[:, iq1:iq2].set( -dlfdlq[None, :] * thetan[:, None] / v[None, :] / kmodes[:, None] / 3.0 )
    y = y.at[:, iq2:iq3].set( -0.5 * dlfdlq[None, :] * shearn[:, None] )
    y = y.at[:, iq3:].set( 0.0 )
    
    return y

@partial(jax.jit, static_argnames=("lmax","lmaxnu","nqmax","rtol","atol"))
def evolve_one_mode( *, y0, tau_start, tau_max, tau_out, param, kmode, lmax, lmaxnu, nqmax, rtol, atol ):

    model = diffrax.ODETerm( 
        lambda tau, y , params : 
            model_synchronous( tau=tau, yin=y, param=param, kmode=kmode, lmax=lmax, lmaxnu=lmaxnu, nqmax=nqmax ) 
    )

    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=tau_out)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol) #, pcoeff=0.4, icoeff=0.3, dcoeff=0)
    sol = diffrax.diffeqsolve(
        terms=model,
        solver=solver,
        t0=tau_start,
        t1=tau_max,
        dt0=tau_start/2,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10010,
    )
    return sol.ys


# @partial(jax.jit, static_argnames=("num_k","lmax","lmaxnu","nqmax","rtol","atol"))
def evolve_perturbations( *, param, aexp_out, kmin : float, kmax : float, num_k : int, lmax : int = 8, lmaxnu : int = 8, nqmax : int = 15, rtol: float = 1e-5, atol: float = 1e-5 ):
    """evolve cosmological perturbations in the synchronous gauge

    Args:
        param (_type_): _description_
        aexp_out (_type_): _description_
        kmin (float): _description_
        kmax (float): _description_
        num_k (int): _description_
        lmax (int, optional): _description_. Defaults to 32.
        lmaxnu (int, optional): _description_. Defaults to 32.
        nqmax (int, optional): _description_. Defaults to 15.
        rtol (float, optional): _description_. Defaults to 1e-3.
        atol (float, optional): _description_. Defaults to 1e-4.

    Returns:
        _type_: _description_
    """
    kmodes = jnp.geomspace(kmin, kmax, num_k)
    nvar   = 7 + 3 * (lmax + 1) + nqmax * (lmaxnu + 1)

    # determine output times from aexp_out
    tau_out = jax.vmap( lambda a: param['tau_of_a_spline'](a) )(aexp_out)
    tau_start = 0.1 #jnp.minimum(1e-3 / jnp.max(kmodes), 0.1)
    tau_max = jnp.max(tau_out)
    nout = aexp_out.shape[0]
    param['nout'] = nout
    
    # set initial conditions
    y0 = adiabatic_ics( tau=tau_start, param=param, kmodes=kmodes, num_k=num_k, nvar=nvar, lmax=lmax, nqmax=nqmax )
    
    # solve ODEs 
    y1 = jax.vmap(
        lambda k_y0 : evolve_one_mode( y0=k_y0[1:], tau_start=tau_start, tau_max=tau_max, tau_out=tau_out, 
                                        param=param, kmode=k_y0[0], lmax=lmax, lmaxnu=lmaxnu, nqmax=nqmax, rtol=rtol, atol=atol ),
                                        in_axes=0
    )(jnp.append(kmodes[:,None],y0,axis=1))
    # else:
    #     y1 = jnp.zeros((num_k,nvar))

    #     for ik,kmode in enumerate(kmodes):
    #         def model_( args ): 
    #             return model_synchronous( tau=args[0], yin=args[1:], param=param, kmode=kmode, lmax=lmax, lmaxnu=lmaxnu, nqmax=nqmax ).flatten()
        
    #         def model( tau, y ): 
    #             return model_synchronous( tau=tau, yin=y, param=param, kmode=kmode, lmax=lmax, lmaxnu=lmaxnu, nqmax=nqmax ).flatten()

    #         def jac_( y ):
    #             return jax.jacfwd( model_ )( y )
            
    #         def jac( tau, y ):
    #             ty = jnp.append(jnp.array([tau]),y)
    #             j = jac_( ty ) 
    #             return jnp.array(j[:,1:])
            
    #         sol = solve_ivp( model, (tau_start, tau_max), y0[ik,:], jac=jac, method='BDF', rtol=rtol, atol=atol )
    #         y1 = y1.at[ik,...].set( sol.y[:,-1] )
    
    return y1, kmodes
    # return sol.ys.reshape((num_k, nvar, nout)), kmodes, sol.ts

