import jax
import jax.numpy as jnp

def lngamma_complex_e( z : complex ):
  """Log[Gamma(z)] for z complex, z not a negative integer Uses complex Lanczos method. Note that the phase part (arg)
    is not well-determined when |z| is very large, due to inevitable roundoff in restricting to (-Pi,Pi].
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


