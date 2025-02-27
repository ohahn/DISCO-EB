import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class spline_interpolation(object):
    def __init__(self, xin: jnp.ndarray, yin: jnp.ndarray, integrate_from_start: bool = True):
        # def spline_interpolation(x: jnp.ndarray, y: jnp.ndarray, integrate_from_start: bool = True):
        """
        Constructs a natural cubic spline interpolator and an integrator from input data x and y.
        
        The spline is built by solving a tridiagonal system for the second derivatives
        (using the Thomas algorithm) under natural boundary conditions (S[0] = S[-1] = 0).
        
        In addition to returning an interpolator function that evaluates the spline at new x values,
        this function returns an integrator function that computes the definite integral of the spline.
        The integrator takes an extra flag:
        - from_start=True returns the integral from x[0] to x_new.
        - from_start=False returns the integral from x_new to x[-1].
        
        Args:
            x: 1D array of x-coordinates (assumed sorted in increasing order).
            y: 1D array of y-coordinates.
        
        Returns:
            A tuple (interpolator, integrator) where:
            - interpolator(x_new) returns the spline value(s) at x_new.
            - integrator(x_new, from_start=True) returns the integral from x[0] to x_new,
                or, if from_start=False, the integral from x_new to x[-1].
        """
        # filter out NaNs at end of arrays
        def _fill_forward( last_observed_yi, yi, fac ):
            yi = jnp.where(jnp.isnan(yi), last_observed_yi*fac, yi)
            return yi, yi
        
        # n = jnp.sum(jnp.isnan(yin) == False)
        n = yin.shape[0]
        _, y = jax.lax.scan(lambda c,v: _fill_forward(c,v,1.0), yin[0], yin)
        _, x = jax.lax.scan(lambda c,v: _fill_forward(c,v,1.01), xin[0], xin)

        
        if n < 2:
            raise ValueError("There must be at least two data points.")
        
        # Compute intervals between knots.
        h = x[1:] - x[:-1]  # shape: (n-1,)
        
        # Compute the second derivatives S at the knots using a Thomas algorithm.
        m = n - 2  # number of interior points
        if m > 0:
            # Build tridiagonal system for S[1] ... S[n-2]
            a = h[:-1]              # lower diagonal (length m)
            b_diag = 2 * (h[:-1] + h[1:])  # main diagonal (length m)
            c = h[1:]               # upper diagonal (length m)
            d = 6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
            
            # Allocate arrays for modified coefficients.
            cp = jnp.zeros(m)
            dp = jnp.zeros(m)
            cp = cp.at[0].set(c[0] / b_diag[0])
            dp = dp.at[0].set(d[0] / b_diag[0])
            
            def forward_step(i, carry):
                cp, dp = carry
                denom = b_diag[i] - a[i] * cp[i - 1]
                cp = cp.at[i].set(jnp.where(i < m - 1, c[i] / denom, 0.0))
                dp = dp.at[i].set((d[i] - a[i] * dp[i - 1]) / denom)
                return (cp, dp)
            
            cp, dp = jax.lax.fori_loop(1, m, forward_step, (cp, dp))
            
            # Backward substitution.
            S_interior = jnp.zeros(m)
            S_interior = S_interior.at[m - 1].set(dp[m - 1])
            
            def backward_step(i, S_int):
                idx = m - 2 - i
                S_int = S_int.at[idx].set(dp[idx] - cp[idx] * S_int[idx + 1])
                return S_int
            
            S_interior = jax.lax.fori_loop(0, m - 1, backward_step, S_interior)
            
            # Natural boundary conditions: S[0] = S[-1] = 0.
            S_full = jnp.concatenate([jnp.array([0.]), S_interior, jnp.array([0.])])
        else:
            # Only two points – linear interpolation.
            S_full = jnp.array([0., 0.])
        
        # Precompute coefficients and integrals on each interval.
        # For interval i, the cubic polynomial is represented as:
        #   P_i(x) = a_i + b_i*(x - x[i]) + c_i*(x - x[i])^2 + d_i*(x - x[i])^3
        # where:
        #   a_i = y[i]
        #   b_i = (y[i+1]-y[i])/h[i] - (h[i]/6)*(S_full[i+1]+2*S_full[i])
        #   c_i = S_full[i] / 2
        #   d_i = (S_full[i+1]-S_full[i]) / (6*h[i])
        a_i = y[:-1]
        b_i = (y[1:] - y[:-1]) / h - h * (S_full[1:] + 2 * S_full[:-1]) / 6.0
        c_i = S_full[:-1] / 2.0
        d_i = (S_full[1:] - S_full[:-1]) / (6.0 * h)
        
        # The exact integral over the full interval [x[i], x[i+1]] is:
        #   I_full[i] = a_i*h[i] + b_i*h[i]^2/2 + c_i*h[i]^3/3 + d_i*h[i]^4/4
        I_full = a_i * h + b_i * (h ** 2) / 2 + c_i * (h ** 3) / 3 + d_i * (h ** 4) / 4
        # Cumulative integral from x[0] up to each knot.
        I_cum = jnp.concatenate([jnp.array([0.]), jnp.cumsum(I_full)])
        I_total = I_cum[-1]
        
        # Store the spline data.
        self._n_ = n
        self._x_, self._y_, self._integrate_from_start_ = x, y, integrate_from_start
        self._S_full_, self._I_total_, self._I_cum_ = S_full, I_total, I_cum

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = (self._x_, self._y_, self._integrate_from_start_, (self._S_full_, self._I_total_, self._I_cum_))
        # The leaves (child nodes) are the JAX arrays that we want to be traced.
        # Any auxiliary static data can be put in aux_data.
        aux_data = {'n': self._n_}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)  # Create instance without calling __init__
        obj._x_, obj._y_, obj._integrate_from_start_, (obj._S_full_, obj._I_total_, obj._I_cum_) = children
        obj._n_ = aux_data['n']
        return obj

    def evaluate(self, x_new: jnp.ndarray):
        """
        Evaluates the natural cubic spline at new x positions.
        
        Args:
            x_new: scalar or 1D array of new x values.
            
        Returns:
            Interpolated y values.
        """
        n = self._x_.shape[0]
        # x_new = jnp.atleast_1d(x_new)
        # Find the interval index i such that x[i] <= x_new < x[i+1]
        idx = jnp.clip(jnp.searchsorted(self._x_, x_new) - 1, 0, n - 2)
        h_local = self._x_[idx + 1] - self._x_[idx]
        d_val = x_new - self._x_[idx]  # local offset
        t = d_val / h_local     # normalized coordinate
        A = 1 - t
        B = t
        # Standard cubic spline evaluation.
        y_new = (A * self._y_[idx] + B * self._y_[idx + 1] +
                 ((A ** 3 - A) * self._S_full_[idx] + (B ** 3 - B) * self._S_full_[idx + 1]) *
                 (h_local ** 2) / 6.0)
        return y_new #jnp.where(x_new.shape[0] == 1, y_new[0], y_new)
    
    def integral(self, x_new: jnp.ndarray):
        """
        Evaluates the definite integral of the spline.
        
        Args:
            x_new: scalar or 1D array of new x values.
            from_start: if True, returns the integral from x[0] to x_new;
                        if False, returns the integral from x_new to x[-1].
                        
        Returns:
            The definite integral values.
        """
        n = self._x_.shape[0]
        x_new = jnp.atleast_1d(x_new)
        # Locate the interval index for each x_new.
        idx = jnp.clip(jnp.searchsorted(self._x_, x_new) - 1, 0, n - 2)
        h_local = self._x_[idx + 1] - self._x_[idx]
        d_val = x_new - self._x_[idx]
        # Compute the local coefficients for the interval.
        a_local = self._y_[idx]
        b_local = (self._y_[idx + 1] - self._y_[idx]) / h_local - h_local * (self._S_full_[idx + 1] + 2 * self._S_full_[idx]) / 6.0
        c_local = self._S_full_[idx] / 2.0
        d_local = (self._S_full_[idx + 1] - self._S_full_[idx]) / (6.0 * h_local)
        # Compute the partial integral over the interval from x[idx] to x_new:
        I_partial = (a_local * d_val +
                     b_local * d_val ** 2 / 2 +
                     c_local * d_val ** 3 / 3 +
                     d_local * d_val ** 4 / 4)
        I_forward = self._I_cum_[idx] + I_partial
        # If integrating from the start, return I_forward; otherwise, subtract from total.
        if self._integrate_from_start_:
            result = jnp.where(x_new.shape[0] == 1, I_forward[0], I_forward)
        else:
            result = jnp.where(x_new.shape[0] == 1, self._I_total_ - I_forward[0], self._I_total_ - I_forward)
        return result
    
    def derivative(self, x_new: jnp.ndarray):
        """
        Computes the derivative of the spline at new x positions.
        
        Args:
            x_new: scalar or 1D array of new x values.
            
        Returns:
            The derivative (dy/dx) evaluated at x_new.
        """
        n = self._x_.shape[0]
        x_new = jnp.atleast_1d(x_new)
        idx = jnp.clip(jnp.searchsorted(self._x_, x_new) - 1, 0, n - 2)
        h_local = self._x_[idx + 1] - self._x_[idx]
        d_val = x_new - self._x_[idx]
        # Local coefficients (as defined in the cubic polynomial):
        b_local = (self._y_[idx + 1] - self._y_[idx]) / h_local - h_local * (self._S_full_[idx + 1] + 2 * self._S_full_[idx]) / 6.0
        c_local = self._S_full_[idx] / 2.0
        d_local = (self._S_full_[idx + 1] - self._S_full_[idx]) / (6.0 * h_local)
        # Derivative of P_i(x) = b_i + 2*c_i*(x-x_i) + 3*d_i*(x-x_i)^2
        dydx = b_local + 2 * c_local * d_val + 3 * d_local * d_val**2
        return jnp.where(x_new.shape[0] == 1, dydx[0], dydx)
    
    def derivative2(self, x_new: jnp.ndarray):
        """
        Computes the derivative of the spline at new x positions.
        
        Args:
            x_new: scalar or 1D array of new x values.
            
        Returns:
            The derivative (dy/dx) evaluated at x_new.
        """
        n = self._x_.shape[0]
        x_new = jnp.atleast_1d(x_new)
        idx = jnp.clip(jnp.searchsorted(self._x_, x_new) - 1, 0, n - 2)
        h_local = self._x_[idx + 1] - self._x_[idx]
        d_val = x_new - self._x_[idx]
        # Local coefficients (as defined in the cubic polynomial):
        b_local = (self._y_[idx + 1] - self._y_[idx]) / h_local - h_local * (self._S_full_[idx + 1] + 2 * self._S_full_[idx]) / 6.0
        c_local = self._S_full_[idx] / 2.0
        d_local = (self._S_full_[idx + 1] - self._S_full_[idx]) / (6.0 * h_local)
        # 2nd derivative of P_i(x) = 2*c_i + 6*d_i*(x-x_i)
        d2ydx2 = 2 * c_local + 6 * d_local * d_val
        return jnp.where(x_new.shape[0] == 1, d2ydx2[0], d2ydx2)

# import jax
# import jax.numpy as jnp
# import equinox as eqx

# @jax.tree_util.register_pytree_node_class
# class spline_interpolation: #(object):
#     def __init__(self, x: jnp.ndarray, y: jnp.ndarray):
#         # Assume x and y are 1D arrays.
#         i = jnp.isnan(y) == False
#         # self.x = x
#         self.x = jnp.where(jnp.isnan(y),jnp.linspace(2*jnp.max(x),10*jnp.max(x),len(x)),x) #jnp.asarray(x[i])
#         self.y = jnp.where(jnp.isnan(y),0.0,y) #jnp.asarray(y[i])
#         self.n = self.x.shape[0]
#         if self.n < 2:
#             raise ValueError("Need at least two points.")

#         # Compute step sizes
#         self.h = self.x[1:] - self.x[:-1]

#         # Compute second derivatives using the Thomas algorithm
#         self.S = self._compute_S(self.x, self.y)

#         # Precompute cubic coefficients for each interval [x_i, x_{i+1}]:
#         # a = y[i]
#         # b = (y[i+1]-y[i])/h[i] - (S[i+1] + 2*S[i])*h[i]/6
#         # c = S[i]/2
#         # d = (S[i+1]-S[i])/(6*h[i])
#         self.a_coef = self.y[:-1]
#         self.b_coef = (self.y[1:] - self.y[:-1]) / self.h - (self.S[1:] + 2 * self.S[:-1]) * self.h / 6.0
#         self.c_coef = self.S[:-1] / 2.0
#         self.d_coef = (self.S[1:] - self.S[:-1]) / (6.0 * self.h)

#         # Precompute cumulative integrals over intervals for integration.
#         I_full = (self.a_coef * self.h +
#                   self.b_coef * self.h**2 / 2 +
#                   self.c_coef * self.h**3 / 3 +
#                   self.d_coef * self.h**4 / 4)
#         self.I_cum = jnp.concatenate([jnp.array([0.]), jnp.cumsum(I_full)])
#         self.I_total = self.I_cum[-1]

#     def _compute_S(self, x, y):
#         n = x.shape[0]
#         m = n - 2  # interior nodes
#         if m < 1:
#             return jnp.array([0., 0.])
#         h = x[1:] - x[:-1]
#         # Build system: A * S_interior = d
#         a_sys = h[:-1]              # lower diagonal
#         b_sys = 2 * (h[:-1] + h[1:])  # main diagonal
#         c_sys = h[1:]               # upper diagonal
#         d_sys = 6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
        
#         cp = jnp.zeros(m)
#         dp = jnp.zeros(m)
#         cp = cp.at[0].set(c_sys[0] / b_sys[0])
#         dp = dp.at[0].set(d_sys[0] / b_sys[0])
        
#         def forward(i, carry):
#             cp, dp = carry
#             denom = b_sys[i] - a_sys[i] * cp[i - 1]
#             cp = cp.at[i].set(jnp.where(i < m - 1, c_sys[i] / denom, 0.0))
#             dp = dp.at[i].set((d_sys[i] - a_sys[i] * dp[i - 1]) / denom)
#             return (cp, dp)
        
#         cp, dp = jax.lax.fori_loop(1, m, forward, (cp, dp))
        
#         S_interior = jnp.zeros(m)
#         S_interior = S_interior.at[m - 1].set(dp[m - 1])
        
#         def backward(i, S_int):
#             idx = m - 2 - i
#             S_int = S_int.at[idx].set(dp[idx] - cp[idx] * S_int[idx + 1])
#             return S_int
        
#         S_interior = jax.lax.fori_loop(0, m - 1, backward, S_interior)
#         S_full = jnp.concatenate([jnp.array([0.]), S_interior, jnp.array([0.])])
#         return S_full

#     @eqx.filter_jit
#     def evaluate(self, x_new: jnp.ndarray):
#         """Evaluates the spline at new x positions."""
#         # x_new = jnp.atleast_1d(x_new)
#         idx = jnp.clip(jnp.searchsorted(self.x, x_new) - 1, 0, self.n - 2)
#         delta = x_new - self.x[idx]
#         y_new = (self.a_coef[idx] + delta * (self.b_coef[idx] + delta * (self.c_coef[idx] + delta * self.d_coef[idx])))
#         return y_new 
        
#     def integrate(self, x_new: jnp.ndarray, from_start: bool = True):
#         """Computes the definite integral of the spline.
        
#         If from_start is True, returns ∫[x[0], x_new] f(x) dx,
#         otherwise returns ∫[x_new, x[-1]] f(x) dx.
#         """
#         x_new = jnp.atleast_1d(x_new)
#         idx = jnp.clip(jnp.searchsorted(self.x, x_new) - 1, 0, self.n - 2)
#         delta = x_new - self.x[idx]
#         I_partial = (self.a_coef[idx] * delta +
#                      self.b_coef[idx] * delta**2 / 2 +
#                      self.c_coef[idx] * delta**3 / 3 +
#                      self.d_coef[idx] * delta**4 / 4)
#         I_forward = self.I_cum[idx] + I_partial
#         result = jnp.where(x_new.shape[0] == 1,
#                            I_forward[0] if from_start else self.I_total - I_forward[0],
#                            jnp.where(from_start, I_forward, self.I_total - I_forward))
#         return result

#     def derivative(self, x_new: jnp.ndarray):
#         """Computes the derivative of the spline at new x positions."""
#         x_new = jnp.atleast_1d(x_new)
#         idx = jnp.clip(jnp.searchsorted(self.x, x_new) - 1, 0, self.n - 2)
#         delta = x_new - self.x[idx]
#         d_new = self.b_coef[idx] + 2 * self.c_coef[idx] * delta + 3 * self.d_coef[idx] * delta**2
#         return jnp.where(x_new.shape[0] == 1, d_new[0], d_new)

#     # --- PyTree Methods ---
#     def tree_flatten(self):
#         # The leaves (child nodes) are the JAX arrays that we want to be traced.
#         children = (self.x, self.y, self.S, self.a_coef, self.b_coef,
#                     self.c_coef, self.d_coef, self.h, self.I_cum, self.I_total)
#         # Any auxiliary static data can be put in aux_data.
#         aux_data = {'n': self.n}
#         return children, aux_data

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         # Reconstruct the object from its children and aux_data.
#         (x, y, S, a_coef, b_coef, c_coef, d_coef, h, I_cum, I_total) = children
#         obj = cls.__new__(cls)  # Create instance without calling __init__
#         obj.x = x
#         obj.y = y
#         obj.S = S
#         obj.a_coef = a_coef
#         obj.b_coef = b_coef
#         obj.c_coef = c_coef
#         obj.d_coef = d_coef
#         obj.h = h
#         obj.I_cum = I_cum
#         obj.I_total = I_total
#         obj.n = aux_data['n']
#         return obj
