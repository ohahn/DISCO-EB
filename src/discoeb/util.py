import jax
import jax.numpy as jnp

def lngamma_complex_e( z : complex ):
  """Log[Gamma(z)] for z complex, z not a negative integer Uses complex Lanczos method. Note that the phase part (arg)
    is not well-determined when `|z|` is very large, due to inevitable roundoff in restricting to (-Pi,Pi].
    -- adapted from GSL --
 
   Args:
      z (complex): input value

  Returns:
      complex: lnr + i arg, where lnr = log|Gamma(z)|, arg = arg(Gamma(z))  in (-Pi, Pi]
  """
  def lngamma_lanczos_complex( z : complex ):
    # Lanzcos approximation [J. SIAM Numer. Anal, Ser. B, 1 (1964) 86]
    lanczos_7_c = jnp.array([
      0.99999999999980993227684700473478,
      676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
      771.3234287776530788486528258894,
    -176.61502916214059906584551354,
      12.507343278686904814458936853,
    -0.13857109526572011689554707,
      9.984369578019570859563e-6,
      1.50563273514931155834e-7
    ])
    LogRootTwoPi_ = 0.9189385332046727418
    z  = z - 1.0
    Ag = lanczos_7_c[0] + jnp.sum( lanczos_7_c[1:] / jnp.abs(z+jnp.arange(1,9))**2 * jnp.conj(z+jnp.arange(1,9)))
    return (z+0.5)*jnp.log(z+7.5) - (z+7.5) + LogRootTwoPi_ + jnp.log(Ag)
  
  lnpi = 1.14472988584940017414342735135
  return jax.lax.cond( jnp.real(z) <= 0.5, 
                      lambda zz: lnpi - jnp.log( jnp.sin(jnp.pi*zz) ) - lngamma_lanczos_complex(1.0-zz), 
                      lambda zz: lngamma_lanczos_complex(zz), z )


def spherical_bessel( lmax, x ):
  """
  Compute spherical Bessel functions jn(x) and their derivatives.

  Parameters
  ----------
  lmax : int
    Maximum order of the Bessel functions, lmax>=0.
  x : float
    Argument at which the Bessel function is evaluated.

  Returns
  -------
  jn : NDArray of float, shape (lmax+1,2,len(x))
    Values of the Bessel function and derivatives for each order and for all x
  """
  x = jnp.atleast_1d(x)

  assert lmax >= 0, "spherical_bessel: lmax must be >= 0"

  res = jax.vmap( lambda xx: jnp.asarray(spherical_bessel_( lmax, xx )) )( x )

  return res


def spherical_bessel_( n : int, x : float ):
  """
  Compute spherical Bessel functions jn(x) and their derivatives.

  Parameters
  ----------
  n : int
    Maximum order of the Bessel functions.
  x : float
    Argument at which the Bessel functions jn are evaluated.

  Returns
  -------
  jn : float
    Values of the Bessel function.
  djn : float
    Value of the derivative of the Bessel function.

  """
  nm = n
  sj = jnp.zeros(n+1)
  dj = jnp.zeros(n+1)

  eps = jnp.sqrt(jnp.finfo(x.dtype).eps)
  tiny = (jnp.finfo(x.dtype).tiny)**(1/3)

  if x.dtype == jnp.float32:
    m1, m2 = 25, 5
  elif x.dtype == jnp.float64:
    m1, m2 = 200, 15


  # n=0: j0(x) = sin(x)/x
  sj = sj.at[0].set( jnp.where( jnp.abs(x) < eps , 1.0-x**2/6, jnp.sin(x) / x ) )
  dj = dj.at[0].set( jnp.where( jnp.abs(x) < eps , -x/3, (jnp.cos(x) - jnp.sin(x) / x) / x ) )

  # n=1: j1(x) = (sin(x) - x cos(x))/x^2
  sj = sj.at[1].set( jnp.where( jnp.abs(x) < eps , x/3 - x**3/30 ,(sj[0] - jnp.cos(x)) / x ) )
  dj = dj.at[1].set( jnp.where( jnp.abs(x) < eps , 1/3 - x**2/10 ,(sj[0]+dj[0]/x + (sj[0]-jnp.cos(x))/x**2) ) )

  # n>=2: recurrence relation
  if n >= 2:
    
    x = jnp.maximum(x, tiny)

    sa = sj[0]
    sb = sj[1]
    m  = msta1( x, m1 )
    nm = jnp.where(m < n, m, nm )
    m  = jnp.where(m < n, m, msta2(x, n, m2 ))

    def condition(state):
        _, _, _, _, k = state
        return k >= 0

    def body(state):
        facc, f0, f1, x, k = state
        f = (2.0 * k + 3.0) * f1 / x - f0
        # Update the state with the new values of f0, f1, and decrement k
        facc = facc.at[k].set( f )
        return facc, f1, f, x, k - 1
    
    f_values = jnp.zeros( n + 1 )
    f_values, final_f0, final_f1, _, _ = jax.lax.while_loop( condition, body, (f_values, 0.0, tiny, x, m) )

    cs = jnp.zeros_like( sa )
    cs = jnp.where( jnp.abs(sa) > jnp.abs(sb), sa / final_f1, cs )
    cs = jnp.where( jnp.abs(sa) <= jnp.abs(sb), sb / final_f0, cs ) 
    
    k_range = jnp.arange(n, 1, -1)
    sj = sj.at[k_range].set(cs * f_values[k_range])

    k_range = jnp.arange(1, n + 1)
    dj = dj.at[k_range].set( (sj[k_range-1] - (k_range + 1.0)) * sj[k_range] / x )

  return sj, dj


def envj( n : int, x : float ) -> float:
  return jnp.log10(6.28*n)/2 - n*jnp.log10(1.36*x/n)


def msta1(x, mp ):
  """
  spherical_bessel helper function: determine the starting point for backward recurrence such 
  that the magnitude of Jn(x) at that point is about 10^(-MP).

  Parameters
  ----------
  x : float
    Argument of jn(x).
  mp : float
    Value of magnitude.

  Returns
  -------
  int
    Starting point.

  """
  a0 = jnp.abs(x)
  n0 = (1.1 * a0).astype(int) + 1
  f0 = envj(n0, a0) - mp
  n1 = n0 + 5
  f1 = envj(n1, a0) - mp

  def cond_fun(loop_vars):
      n0, n1, _, _, it = loop_vars
      return (jnp.abs(n0-n1) >= 1) & (it < 20)

  def body_fun(loop_vars):
      n0, n1, f0, f1, it = loop_vars
      nn = (n1 - (n1 - n0) / (1.0 - f0 / f1)).astype(int)
      f = envj(nn, a0) - mp
      n0_update = n1
      f0_update = f1
      n1_update = nn
      f1_update = f
      return (n0_update, n1_update, f0_update, f1_update, it + 1)

  initial_vars = (n0, n1, f0, f1, 0)
  _, nn_final, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_vars)

  return nn_final


def msta2( x : float, n : int, mp : float ):
  """
  spherical_bessel helper function: determine the starting point for backward recurrence 
  such that all Jn(x) has MP significant digits.

  Parameters
  ----------
  x : float
    Argument of jn(x).
  n : int
    Order of jn(x).
  mp : float
    Significant digit.

  Returns
  -------
  int
    Starting point.

  """
  if x.dtype == jnp.float32:
    maxit = 10
  else:
    maxit = 20

  a0 = jnp.abs(x)
  hmp = 0.5 * mp
  ejn = envj(n, a0)
  obj = jnp.where(ejn <= hmp, mp, hmp + ejn)
  n0  = jnp.where(ejn <= hmp, (1.1 * a0).astype(int) + 1, n)

  f0 = envj(n0, a0) - obj
  n1 = n0 + 5
  f1 = envj(n1, a0) - obj

  def cond_fun(vars):
      n0, n1, _, _, iter_count = vars
      return (jnp.abs(n0-n1) >= 1) & (iter_count < maxit)

  def body_fun(vars):
      n0, n1, f0, f1, iter_count = vars
      nn = (n1 - (n1 - n0) / (1.0 - f0 / f1)).astype(int)
      f = envj(nn, a0) - obj
      return (n1, nn, f1, f, iter_count + 1)

  initial_vars = (n0, n1, f0, f1, 0)
  _, nn_final, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_vars)

  return nn_final + 10




def root_find_bisect( *, func, xleft, xright, numit, param ):
  """
  Simple bisection routine for root finding.
  
  Parameters
  ----------
  func : function
    Function to be evaluated.
  xleft : float, jax.Array
    Left boundary of the interval.
  xright : float, jax.Array
    Right boundary of the interval.
  numit : int
    Number of iterations.

  Returns
  -------
  x0 : float
    Approximation to the root, given by the midpoint of the final interval.

  """
  for i in range(numit):
        xmid = 0.5 * (xleft + xright)
        xleft, xright = jax.lax.cond(func(xmid, param) * func(xleft, param) > 0, lambda x : (xmid, xright), lambda x : (xleft, xmid), None )
  # def body(carry, i):
  #   xleft, xright, param = carry
  #   xmid = 0.5 * (xleft + xright)
  #   carry = jax.lax.cond(func(xmid, param) * func(xleft, param) > 0, lambda x : (xmid, xright, param), lambda x : (xleft, xmid, param), None )
  #   return carry, xmid

  # (xleft, xright, param), _ = jax.lax.scan(body, (xleft, xright, param), jnp.arange(numit))
  return 0.5 * (xleft + xright)


def softclip(x, a_min, a_max):
    """
    Softclip function that is strictly monotonous.
    """
    y = jnp.clip(x, a_min, a_max)
    return jnp.where(x < a_min, a_min + jnp.log(jnp.abs(x - a_min) + 1) * jnp.sign(x - a_min), jnp.where(x > a_max, a_max + jnp.log(jnp.abs(x - a_max) + 1) * jnp.sign(x - a_max), y))


def savgol_filter( *, y : jax.Array, window_length : int|jax.Array, polyorder : int ) -> jax.Array:
  """
  Apply a Savitzky-Golay filter to an array.

  Parameters
  ----------
  y : jax.Array
    Array to be filtered.
  window_length : int, jax.Array
    Length of the filter window.
  polyorder : int
    Order of the polynomial to fit to each window.

  Returns
  -------
  jax.Array
    Filtered array.

  """
  # compute the coefficients of the filter
  halflen, rem = divmod(window_length, 2)
  if rem == 0:
      pos = halflen - 0.5
  else:
      pos = halflen

  x = jnp.arange(-pos, window_length - pos, dtype=float)[::-1]

  order = jnp.arange(polyorder + 1).reshape(-1, 1)
  A = x ** order

  Y = jnp.zeros(polyorder + 1)
  Y = Y.at[0].set( 1.0 )

  # Find the least-squares solution of A*c = y
  coeffs, _, _, _ = jnp.linalg.lstsq(A, Y)

  return jnp.convolve(y,coeffs,mode='same')

