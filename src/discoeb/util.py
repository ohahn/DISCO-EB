import jax
import jax.numpy as jnp

def lngamma_complex_e( z : complex ):
  """Log[Gamma(z)] for z complex, z not a negative integer Uses complex Lanczos method. Note that the phase part (arg)
    is not well-determined when ``|z|`` is very large, due to inevitable roundoff in restricting to (-Pi,Pi].
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


def gauss_laguerre_weights( n : int ) -> tuple[jax.Array, jax.Array]:
    """
    Compute the nodes and weights for n-point Gauss-Laguerre quadrature
    for the weight function exp(-x) on [0, ∞).

    Parameters:
        n : int
            Number of quadrature points.
    
    Returns:
        nodes : ndarray
            The quadrature nodes (abscissae).
        weights : ndarray
            The quadrature weights.
    """
    # Diagonal entries: a_i = 2*i - 1, for i = 1,...,n
    i = jnp.arange(1, n+1)
    a = 2*i - 1

    # Off-diagonal entries: b_i = i for i = 1,..., n-1.
    b = jnp.arange(1, n)
    
    # Construct the symmetric tridiagonal Jacobi matrix.
    J = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)
    
    # Compute eigenvalues and eigenvectors.
    nodes, eigenvectors = jnp.linalg.eigh(J)
    
    # The weights are the squares of the first component of each eigenvector.
    # (For Gauss-Laguerre, μ₀ = ∫₀∞ e^(–x) dx = 1.)
    weights = eigenvectors[0, :]**2
    
    return nodes, weights


def generalized_gauss_laguerre_weights(n, alpha):
    """
    Compute nodes and weights for n-point generalized Gauss-Laguerre quadrature,
    which approximates integrals of the form
        ∫₀∞ x^α f(x) e^(-x) dx.
    
    Parameters:
        n : int
            Number of quadrature points.
        alpha : float
            The parameter in the weight function x^α e^(-x).
    
    Returns:
        nodes : ndarray
            The quadrature nodes (abscissae).
        weights : ndarray
            The quadrature weights.
    """
    # Indices i = 1, 2, ..., n.
    i = jnp.arange(1, n+1)
    
    # Diagonal entries: a_i = 2i - 1 + alpha.
    a = 2*i - 1 + alpha
    
    # Off-diagonal entries for i = 1, ..., n-1: b_i = sqrt(i*(i+alpha))
    i_off = jnp.arange(1, n)
    b = jnp.sqrt(i_off * (i_off + alpha))
    
    # Construct the symmetric tridiagonal Jacobi matrix.
    J = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)
    
    # Compute eigenvalues (nodes) and eigenvectors.
    nodes, V = jnp.linalg.eigh(J)
    
    # The weights are given by the square of the first component of the eigenvectors,
    # multiplied by the zeroth moment: Γ(α+1).
    if type(alpha) == int:
      weights = (V[0, :]**2) * jax.scipy.special.factorial(alpha)
    else:
      weights = (V[0, :]**2) * jax.scipy.special.gamma(alpha + 1)
    
    return nodes, weights


def continuedfraction_Jratio(l, x, niter=5):
  """
  Compute the ratio J_{nu-1}(x)/J_{nu}(x) using the continued fraction (eq. 
    10.10.1 of https://dlmf.nist.gov/10.10) with a fixed number of iterations.

  Parameters
  ----------
  l : float
    Order of the Bessel function.
  x : float
    Argument of the Bessel function.
  niter : int
    Number of iterations for the continued fraction.

  Returns
  -------
  float
    The ratio J_{l-1}(x)/J_{l}(x).
  """
  f0 = 2.0 * l / x
  C0 = f0 
  D0 = jnp.zeros_like(x) 

  def body_fn(carry, _):
    f0, C0, D0 = carry
    j = _ + 1
    bj = (-1.0)**j * 2.0 * (l + j) / x
    Cj = bj + 1.0 / C0
    Dj = 1.0 / (bj + D0)
    fj = f0 * Cj * Dj
    return (fj, Cj, Dj), None

  (f_final, _, _), _ = jax.lax.scan(body_fn, (f0, C0, D0), jnp.arange(niter))
  return f_final


# @partial(jax.jit, static_argnames=['lmax','niterfrac'])
def spherical_bessel(lmax, x, niterfrac=5):
    """
    Spherical Bessel function computation (GPU-optimized), uses forward recurrence formula 
    (eq. 10.6.1 of https://dlmf.nist.gov/10.6) for maximum parallel efficiency, which is 
    however unstable for |x| < \ell, so there the continued fraction expansion (eq. 
    10.10.1 of https://dlmf.nist.gov/10.10) is used instead. Current performance bottleneck
    is that the continued fraction has to be evluated.
    
    Parameters
    ----------
    lmax : int
      The maximum order of the spherical Bessel functions to compute.
    x : array-like
      Input values for which the spherical Bessel functions are computed. Can be a scalar or an array.
    niterfrac : int, optional
      Number of iterations for the continued fraction expansion, by default 5.
    Returns
    -------
    jnp.ndarray
      A stacked array of shape (2, n_x, lmax + 1), where the first sub-array contains the 
      spherical Bessel functions `j_l(x)` for orders `l = 0, ..., lmax`, and the second 
      sub-array contains their derivatives `j'_l(x)`.
    """
    x = jnp.atleast_1d(x)
    n_x = x.shape[0]
    
    # Constants
    eps = jnp.sqrt(jnp.finfo(x.dtype).eps)
    tiny = (jnp.finfo(x.dtype).tiny)**(1/3)
    
    x_abs = jnp.abs(x)
    small_x_mask = x_abs < eps
    
    # Pre-allocate output arrays
    sj = jnp.zeros((n_x, lmax + 1))
    dj = jnp.zeros((n_x, lmax + 1))
    
    # Compute trigonometric functions once
    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)
    
    # j0(x) = sin(x)/x
    j0_small = 1.0 - x**2/6
    j0_large = sin_x / x
    sj = sj.at[:, 0].set(jnp.where(small_x_mask, j0_small, j0_large))
    
    # dj0(x)
    dj0_small = -x/3
    dj0_large = (cos_x - sin_x / x) / x
    dj = dj.at[:, 0].set(jnp.where(small_x_mask, dj0_small, dj0_large))
    
    if lmax >= 1:
        # j1(x) = (sin(x) - x*cos(x))/x^2
        j1_small = x/3 - x**3/30
        j1_large = (sj[:, 0] - cos_x) / x
        sj = sj.at[:, 1].set(jnp.where(small_x_mask, j1_small, j1_large))
        
        # dj1(x)
        dj1_small = 1/3 - x**2/10
        dj1_large = sj[:, 0] + dj[:, 0]/x + (sj[:, 0] - cos_x)/x**2
        dj = dj.at[:, 1].set(jnp.where(small_x_mask, dj1_small, dj1_large))
    
    if lmax >= 2:
        x_safe = jnp.maximum(x_abs, tiny)
        
        # Use forward recurrence for better GPU performance (more parallel)
        # (eq. 10.6.1 of https://dlmf.nist.gov/10.6)
        # j_{n+1} = (2n+1)/x * j_n - j_{n-1}
        def forward_step(n, sj_arr):
            j_next = jnp.where( (x_safe < n-jnp.sqrt(n)), 
                               sj_arr[:, n] / continuedfraction_Jratio(n+1.5, x_safe, niterfrac), 
                               (2.0 * n + 1.0) * sj_arr[:, n] / x_safe - sj_arr[:, n-1] )
            return sj_arr.at[:, n+1].set(j_next)
        
        # Apply forward recurrence
        sj = jax.lax.fori_loop(1, lmax, forward_step, sj)
        
        # Compute derivatives using the recurrence relation (eq. 10.6.2 of https://dlmf.nist.gov/10.6)
        # dj_n = j_{n-1} - (n+1)/x * j_n
        l_indices = jnp.arange(1, lmax + 1)
        x_expanded = x_safe[:, None]
        
        dj_vals = sj[:, l_indices - 1] - (l_indices + 1.0) * sj[:, l_indices] / x_expanded
        dj = dj.at[:, 1:].set(dj_vals)
    
    return jnp.stack([sj, dj], axis=1)


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
