from collections.abc import Callable
from typing import ClassVar, Union, Literal, Optional
from typing_extensions import TypeAlias

import equinox.internal as eqxi
from equinox.internal import ω
import lineax as lx
from jaxtyping import PyTree

import jax.numpy as jnp
import jax

import typing
from typing import Any, TYPE_CHECKING, Union
from jaxtyping import (
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    PyTree,
    Shaped,
)

if TYPE_CHECKING:
    BoolScalarLike = Union[bool, Array, jnp.ndarray]
    FloatScalarLike = Union[float, Array, jnp.ndarray]
    IntScalarLike = Union[int, Array, jnp.ndarray]
elif getattr(typing, "GENERATING_DOCUMENTATION", False):
    # Skip the union with Array in docs.
    BoolScalarLike = bool
    FloatScalarLike = float
    IntScalarLike = int

    #
    # Because they appear in our docstrings, we also monkey-patch some non-Diffrax
    # types that have similar defined-in-one-place, exported-in-another behaviour.
    #

    jtu.Partial.__module__ = "jax.tree_util"

else:
    BoolScalarLike = Bool[ArrayLike, ""]
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]


RealScalarLike = Union[FloatScalarLike, IntScalarLike]

Y = PyTree[Shaped[ArrayLike, "?*y"], "Y"]
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Shaped[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]

DenseInfo = dict[str, PyTree[Array]]
DenseInfos = dict[str, PyTree[Shaped[Array, "times-1 ..."]]]
BufferDenseInfos = dict[str, PyTree[eqxi.MaybeBuffer[Shaped[Array, "times ..."]]]]
sentinel: Any = eqxi.doc_repr(object(), "sentinel")

# from diffrax import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
# from diffrax._heuristics import is_sde

from diffrax import AbstractTerm

def is_sde(terms: PyTree[AbstractTerm]) -> bool:
    return False

from diffrax import LocalLinearInterpolation
from diffrax import RESULTS
from diffrax import AbstractTerm
from diffrax import AbstractAdaptiveSolver

_SolverState: TypeAlias = None

class GRKT4(AbstractAdaptiveSolver):
  r""" Generalized linearly implicit Rosenbrock-Wanner method of order 4 with embedded 3rd order error control.

  Method is the 'GRK4A/T' method from Kaps & Rentrop (1979), retrievable from http://numerik.mi.fu-berlin.de/wiki/WS_2021/NumericsII_Dokumente/Kaps-Rentrop1979_Article_GeneralizedRunge-KuttaMethodsO.pdf
  original article: https://link.springer.com/article/10.1007/BF01396495

  """

  scan_kind: Union[None, Literal["lax", "checkpointed", "bounded"]] = None

  term_structure: ClassVar = AbstractTerm

  # jac_f: Callable = None
  linear_solver: lx.AbstractLinearSolver = lx.LU
  init_later_state: Optional[PyTree] = None

  interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation # TODO: bump to 3rd order Hermite
  
  def order(self, terms):
    return 4
  
  def error_order(self, terms):
    if is_sde(terms):
      return None
    else:
      return 3
  
  def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None
        
  def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        # del made_jump
        control = terms.contr(t0, t1)

        if False:
          #GRK4A variant
          gamma = 0.395
          gamma21 = -0.767672395484
          gamma31 = -0.851675323742; gamma32 = 0.522967289188
          gamma41 = 0.288463109545;  gamma42 = 0.880214273381e-1; gamma43 = -0.337389840627
          
          alpha21 = 0.438
          alpha31 = 0.796920457938; alpha32 = 0.730795420615e-1

          chat1 = 0.346325833758; chat2 = 0.285693175712; chat3 = 0.367980990530
          c1 = 0.199293275701; c2 = 0.482645235674; c3 = 0.680614886256e-1; c4 = 0.25

        else:
          #GRK4T variant
          gamma   = 0.231
          gamma21 = -0.270629667752
          gamma31 = 0.311254483294; gamma32 = 0.00852445628482
          gamma41 = 0.282816832044; gamma42 = -0.457959483281; gamma43 =-0.111208333333

          alpha21 = 0.462
          alpha31 = -0.0815668168327; alpha32 = 0.961775150166
          #alpha41 = alpha31; alpha42 = alpha32; alpha43 = 0.0 # not needed since duplicate coefficients
          chat1   = -0.717088504499; chat2 = 1.77617912176; chat3 = -0.0590906172617
          c1      = 0.217487371653; c2 = 0.486229037990; c3 = 0.; c4 = 0.296283590357

        # Jacobian for all stages
        J = lx.JacobianLinearOperator(lambda y,arg : terms.vf_prod(t0, y, arg, control), y0, args)
        J = lx.linearise(J)
        n = y0.shape[0]

        # compute linear inverse, perform LU decomp
        I  = jnp.eye(n)
        A  = jax.lax.stop_gradient(I - gamma * J.as_matrix())

        LU_and_piv = jax.scipy.linalg.lu_factor(A, overwrite_a=True, check_finite=False)

        # Stage 1
        b1 = terms.vf_prod(t0, y0, args, control)
        k1 = jax.scipy.linalg.lu_solve( LU_and_piv, b1 )
        
        # Stage 2
        t1, y1 = t0, (y0**ω + alpha21 * k1**ω).ω
        f1 = terms.vf_prod(t1, y1, args, control)
        Jk1 = J.mv(k1)
        b2 = (f1**ω + gamma21 * Jk1**ω).ω
        k2 = jax.scipy.linalg.lu_solve( LU_and_piv, b2 )

        # Stage 3
        t2, y2 = t0, (y0**ω + alpha31 * k1**ω + alpha32 * k2**ω).ω
        f2 = terms.vf_prod(t2, y2, args, control)
        Jk2 = J.mv(k2)
        b3 = (f2**ω + gamma31 * Jk1**ω + gamma32 * Jk2**ω).ω
        k3 = jax.scipy.linalg.lu_solve( LU_and_piv, b3 ) 
        
        # Stage 4
        # t3, y3 = t2, y2
        f3 = f2
        Jk3 = J.mv(k3)
        b4 = (f3**ω + gamma41 * Jk1**ω + gamma42 * Jk2**ω + gamma43 * Jk3**ω).ω
        k4 = jax.scipy.linalg.lu_solve( LU_and_piv, b4 )
        
        # Advance Solution
        y1 = (y0**ω + c1 * k1**ω + c2 * k2**ω + c3 * k3**ω + c4 * k4**ω).ω
        y_error = ((c1-chat1)*k1**ω + (c2-chat2)*k2**ω + (c3-chat3)*k3**ω + (c4-0.0)*k4**ω).ω

        dense_info = dict(y0=y0, y1=y1)

        return y1, y_error, dense_info, solver_state, RESULTS.successful


  def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)