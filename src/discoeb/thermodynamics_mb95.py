import jax.lax
import jax.numpy as jnp
from typing import Tuple


def ionize(tempb: float, a: float, adot: float, dtau: float, xe: float, YHe: float, H0: float, Omegab: float) -> float:
    # ... switch for fully implicit (switch=1.0) or semi implicit (switch=0.5);
    iswitch = 0.5

    tion = 1.5789e5  # ionization temperature
    beta0 = 43.082  # ionization coefficient (?)
    dec2g = 8.468e14  # two photon decay rate (in 1/Mpc)

    # recombination coefficient (in sqrt(K)/Mpc).
    alpha0 = 2.3866e-6 * (1.0 - YHe) * Omegab * H0**2

    # coefficient for correction of radiative decay (dimensionless)
    crec = 8.0138e-26 * (1.0 - YHe) * Omegab * H0**2

    # recombination and ionization rates.
    phi2 = jnp.maximum(0.448 * jnp.log(tion / tempb), 0.0)
    alpha = alpha0 / jnp.sqrt(tempb) * phi2 / a**3
    beta = tempb * phi2 * jnp.exp(beta0 - tion / tempb)

    # ... Peebles' correction factor
    def peebles_corr():
        cp1 = crec * dec2g * (1.0 - xe) / (a * adot)
        cp2 = crec * tempb * phi2 * jnp.exp(beta0 - 0.25 * tion / tempb) * (1.0 - xe) / (a * adot)
        return (1.0 + cp1) / (1.0 + cp1 + cp2)
    
    cpeebles = jax.lax.cond(tempb <= 200.0, lambda: 1.0, peebles_corr)

    # ... integrate dxe=bb*(1-xe)-aa*xe*xe by averaging rhs at current tau
    # ... (fraction 1-iswitch) and future tau (fraction iswitch).
    aa = a * dtau * alpha * cpeebles
    bb = a * dtau * beta * cpeebles
    b1 = 1.0 + iswitch * bb
    bbxe = bb + xe - (1.0 - iswitch) * (bb * xe + aa * xe * xe)
    rat = iswitch * aa * bbxe / (b1 * b1)
    #...  Prevent roundoff error
    xe = jax.lax.cond(rat < 5e-5,
                      lambda: bbxe / b1 * (1.0 - rat),
                      lambda: b1 / (2.0 * iswitch * aa) * (jnp.sqrt(4.0 * rat + 1.0) - 1.0))
    return xe


def ionHe(tempb: float, a: float, x0: float, x1: float, x2: float, YHe: float, H0: float, Omegab: float,
          enforce_tolerance: bool = False, tol: float = 1e-12, n_iter_max: int = 6) -> Tuple[float, float]:
    """Compute the helium ionization fractions using the Saha equation
    """
    tion1 = 2.855e5
    tion2 = 6.313e5

    # ... constant for electron partition function per baryon
    b0 = 2.150e24 / ((1.0 - YHe) * Omegab * H0**2)

    # ... electron partition function per baryon
    b = b0 * a**3 * tempb * jnp.sqrt(tempb)

    # ... dimensionless right-hand sides in Saha equations
    r1 = 4.0 * b * jnp.exp(-tion1 / tempb)
    r2 = b * jnp.exp(-tion2 / tempb)

    # ... solve coupled equations iteratively
    c = 0.25 * YHe / (1.0 - YHe)
    err = jnp.inf

    def body_fun(i, vals):
        err, xe, x1, x2 = vals
        xe = x0 + c * (x1 + 2.0 * x2)
        x2new = r1 * r2 / (r1 * r2 + xe * r1 + xe * xe)
        x1 = xe * r1 / (r1 * r2 + xe * r1 + xe * xe)
        err = jnp.fabs(x2new - x2)
        return err, xe, x1, x2new

    xe = x0 + c * (x1 + 2.0 * x2)

    # if tolerance shall be enforced: while-loop is used, which is NOT reverse-mode differentiable!
    if enforce_tolerance:
        cond_fun = lambda vals: vals[0] > tol
        out = jax.lax.while_loop(lambda vals: cond_fun(0, vals), body_fun, (err, xe, x1, x2))

    else:
        # for-loop instead
        out = jax.lax.fori_loop(lower=0, upper=n_iter_max, body_fun=body_fun, init_val=(err, xe, x1, x2))

    x1, x2 = out[2], out[3]
    return x1, x2


# @partial(jax.jit, static_argnames=("nthermo",))
def compute_thermo(*, param, nthermo: int):
    # Output: a, adot, tau, tb, xe_raw, xe, xHeI, xHeII, cs2
    tau0    = param['taumin']
    adot0   = param['adotrad']
    Tcmb    = param['Tcmb'] 
    YHe     = param['YHe']
    taumin  = param['taumin']
    taumax  = param['taumax']
    grhom   = param['grhom']
    grhog   = param['grhog']
    grhor   = param['grhor']
    Omegam  = param['Omegam']
    Omegab  = param['Omegab']
    OmegaDE = param['OmegaDE']
    w_DE_0  = param['w_DE_0']
    w_DE_a  = param['w_DE_a']
    Neff    = param['Neff']
    Nmnu    = param['Nmnu']
    H0      = param['H0']
    logrhonu_sp  = param['logrhonu_of_loga_spline']


    thomc0 = 5.0577e-8 * Tcmb**4
    dlntau = jnp.log(taumax / taumin) / (nthermo - 1)

    # ... initial conditions : assume radiation-dominated universe
    
    
    a0 = adot0 * tau0

    tb0 = Tcmb / a0
    xHII0 = 1.0
    xHeII0 = 0.0
    xHeIII0 = 1.0
    xe0 = xHII0 + 0.25 * YHe / (1.0 - YHe) * (xHeII0 + 2.0 * xHeIII0)
    barssc_raw = 9.1820e-14
    barssc = barssc_raw * (1.0 - 0.75 * YHe + (1.0 - YHe) * xe0)
    cs20 = 4.0 / 3.0 * barssc * tb0
    keys = ("a", "adot", "tau", "tb", "xHII", "xe", "xHeII", "xHeIII", "cs2")
    init = dict(a=a0, adot=adot0, tau=tau0, tb=tb0, xHII=xHII0, xe=xe0, xHeII=xHeII0, xHeIII=xHeIII0, cs2=cs20)

    def scan_fun(carry, x):
        a, adot, tau, tb, xHII, xe, xHeII, xHeIII, cs2 = [carry[k] for k in keys]
        i = x

        new_tau = taumin * jnp.exp(i * dlntau)
        dtau = new_tau - tau

        # integrate Friedmann equation using inverse trapezoidal rule.
        new_a = a + adot * dtau

        # rhonu = jnp.exp(logrhonu_sp.evaluate( jnp.log(new_a) ))
        # rhoDE = new_a**(-3*(1+w_DE_0+w_DE_a)) * jnp.exp(3*(new_a-1)*w_DE_a)
    
        # grho = (
        #     grhom * Omegam / new_a
        #     + (grhog + grhor * (Neff + Nmnu * rhonu)) / new_a**2
        #     + grhom * OmegaDE * rhoDE * new_a**2
        #     + grhom * (1-Omegam-OmegaDE) #FIXME: Omegak
        # )
        # new_adot = jnp.sqrt(grho / 3.0) * new_a
        from .background import get_aprimeoa
        new_adot = get_aprimeoa(param=param, aexp=new_a) * new_a

        new_a = a + 2 * dtau / (1.0 / adot + 1.0 / new_adot)

        # ... baryon temperature evolution: adiabatic except for Thomson cooling
        # ... use quadratic solution.
        tg0 = Tcmb / a
        ahalf = 0.5 * (a + new_a)
        adothalf = 0.5 * (adot + new_adot)

        # ... fe = number of free electrons divided by total number of free baryon
        # ... particles (e+p+H+He). Evaluate at timestep i-1 for convenience; if
        # ... more accuracy is required (unlikely) then this can be iterated with
        # ... the solution of the ionization equation
        fe = (1.0 - YHe) * xe / (1.0 - 0.75 * YHe + (1.0 - YHe) * xe)
        thomc = thomc0 * fe / adothalf / ahalf**3
        etc = jnp.exp(-thomc * (new_a - a))
        a2t = a**2 * (tb - tg0) * etc - Tcmb / thomc * (1.0 - etc)
        # ... use 2nd order Taylor if fe is small to avoid numerical problems with single precision
        a2t_expansion = (a - new_a) * Tcmb + a**2 * (tb - tg0) + (0.5*(a-new_a)**2 * Tcmb + a**2 * (a-new_a) * (tb-tg0))*thomc
        a2t = jax.lax.cond( fe < 1e-3, lambda _: a2t_expansion, lambda _: a2t, operand=None)

        # Set new tb
        new_tb = Tcmb / new_a + a2t / new_a**2
        
        # ... integrate ionization equation
        tbhalf = 0.5 * (tb + new_tb)
        new_xHII = ionize(tbhalf, ahalf, adothalf, dtau, xHII, YHe, H0, Omegab)
        new_xHeII, new_xHeIII = ionHe(tb, new_a, new_xHII, xHeII, xHeIII, YHe, H0, Omegab)

        # set new value
        new_xe = new_xHII + 0.25 * YHe / (1.0 - YHe) * (new_xHeII + 2.0 * new_xHeIII)

        # ... baryon sound speed squared (over c^2)
        dtbdla = -2.0 * new_tb - thomc * a2t / new_a
        barssc = barssc_raw * (1.0 - 0.75 * YHe + (1.0 - YHe) * new_xe)
        new_cs2 = barssc * new_tb * (1.0 - dtbdla / new_tb / 3.0)

        new_carry = dict(a=new_a, adot=new_adot, tau=new_tau, tb=new_tb, xHII=new_xHII, xe=new_xe,
                         xHeII=new_xHeII, xHeIII=new_xHeIII, cs2=new_cs2)
        return new_carry, new_carry

    # def scan_manual(f, init, xs, length=None):
    #     if xs is None:
    #         xs = [None] * length
    #     carry = init
    #     ys = []
    #     for x in xs:
    #         carry, y = f(carry, x)
    #         ys.append(y)
    #     return carry, jnp.stack(ys)

    # scan = scan_manual  # for debugging

    scan = jax.lax.scan
    out = scan(scan_fun, init, xs=jnp.arange(1, nthermo))[1]
    out_with_init = {k: jnp.concatenate((jnp.atleast_1d(init[k]), out[k])) for k in keys}

    return out_with_init, param
