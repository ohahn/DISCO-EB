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

from diffrax import AbstractTerm

def is_sde(terms: PyTree[AbstractTerm]) -> bool:
    return False

from diffrax import LocalLinearInterpolation
from diffrax import RESULTS
from diffrax import AbstractTerm
from diffrax import AbstractAdaptiveSolver

_SolverState: TypeAlias = None

#################################################################################################################################################
#################################################################################################################################################

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

        if True:
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
  
#################################################################################################################################################
#################################################################################################################################################

class Rodas5(AbstractAdaptiveSolver):
  r""" RODAS (Rosenbrock for Differential-Algebraic Systems) of order 5 with embedded 4th order error control
     Diploma Thesis of Giovanna DI MARZO, UGenève, 1993, "Méthodes de Rosenbrock d'ordre 5(4) adaptées aux systèmes différentiels-algébriques"

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
    return 5
  
  def error_order(self, terms):
    if is_sde(terms):
      return None
    else:
      return 4
  
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
        
        gamma  = 0.19

        if True: # type 1 from 
          alpha21 = 0.38
          alpha31 = 0.1899188971074152; alpha32 = 0.1979321027247381
          alpha41 = 0.1110729281178426; alpha42 = 0.5456026683145674; alpha43 = -0.1727037026450261
          alpha51 = 0.2329444418850307; alpha52 = 2.5099380960713898E-02; alpha53 = 0.1443314046300300; alpha54 = 5.4672473406183418E-02
          alpha61 = -3.6201017843430883E-02; alpha62 = 4.208448872731939; alpha63 = -7.549674427720996; alpha64 = -0.2076823626400282; alpha65 = 4.585108935472517
          alpha71 = 7.585261698003052; alpha72 = -15.57426208319938; alpha73 = -8.814406895608121; alpha74 = 1.534698996826085; alpha75 = 16.07870828397837; alpha76 = 0.1900000000000000
          alpha81 = 0.4646018839086969; alpha82 = 0.0; alpha83 = -1.720907508837576; alpha84 = 0.2910480220957973; alpha85 = 1.821778861539924; alpha86 = -4.6521258706842056E-02; alpha87 = 0.19

          beta21 = 7.6920774666285364E-03
          beta31 = -5.8129718999580252E-02; beta32 = -6.3251113355141360E-02
          beta41 = 0.7075715596134048; beta42 = -0.5980299539145789; beta43 = 0.5294131505610923
          beta51 = -3.4975026573934865E-02; beta52 = -0.1928476085817357; beta53 = 8.9839586125126941E-02; beta54 = 2.7613185520411822E-02
          beta61 = 7.585261698003052; beta62 = -15.57426208319938; beta63 = -8.814406895608121; beta64 = 1.534698996826085; beta65 = 16.07870828397837
          beta71 = 0.4646018839086969; beta72 = 0.0; beta73 = -1.720907508837576; beta74 = 0.2910480220957973; beta75 = 1.821778861539924; beta76 = -4.6521258706842056E-02
          beta81 = 0.4646018839086969; beta82 = 0.0; beta83 = -1.720907508837576; beta84 = 0.2910480220957973; beta85 = 1.821778861539924; beta86 = -2.6744882930135193E-02; beta87 = -1.9776375776706864E-02
        else:
          alpha21 = 0.38
          alpha31 = 0.1899809632304806; alpha32 = 0.1862153615948063
          alpha41 = 8.3489136955726816E-02; alpha42 = 0.6956787031729650; alpha43 = -0.2715097757633048
          alpha51 = -1.9749954685321001E-02; alpha52 = 1.152768053768262; alpha53 = -0.5817213956064128; alpha54 = -0.1156932636686364
          alpha61 = 0.8344149384246697; alpha62 = 1.434115012766763; alpha63 = -2.746305299598788; alpha64 = 4.636356700988413; alpha65 = -3.158581352581058
          alpha71 = 3.424382668256867; alpha72 = -9.068294703885030; alpha73 = 3.691091270599460; alpha74 = 5.976075526464554; alpha75 = -3.213254761435850; alpha76 = 0.1900000000000000
          alpha81 = 0.3754294878045745; alpha82 = 0.0; alpha83 = -0.1985496379538312; alpha84 = 1.384545678404626; alpha85 = -0.6985967635977041; alpha86 = -5.2828764657665079E-02; alpha87 = 0.19

          beta21 = -3.8421336706631788E-03
          beta31 = -6.5875081269363735E-02; beta32 = -6.3314423598353655E-02
          beta41 = 0.5348790830634264; beta42 = -0.4715707828658806; beta43 = 0.4433131178017218
          beta51 = 0.7302601562834119; beta52 = -0.2532267377201463; beta53 = 0.3774008675499461; beta54 = -9.8347581382224226E-02
          beta61 = 3.424382668256867; beta62 = -9.068294703885030; beta63 = 3.691091270599460; beta64 = 5.976075526464554; beta65 = -3.213254761435850

          beta71 = 0.3754294878045745; beta72 = 0.0; beta73 = -0.1985496379538312; beta74 = 1.384545678404626; beta75 = -0.6985967635977041; beta76 = -5.2828764657665079E-02
          beta81 = 0.3754294878045745; beta82 = 0.0; beta83 = -0.1985496379538312; beta84 = 1.384545678404626; beta85 = -0.6985967635977041; beta86 = -5.1105182319437466E-02; beta87 = -1.7235823382276139E-03
          
        gamma21 = beta21-alpha21
        gamma31 = beta31-alpha31; gamma32 = beta32-alpha32
        gamma41 = beta41-alpha41; gamma42 = beta42-alpha42; gamma43 = beta43-alpha43
        gamma51 = beta51-alpha51; gamma52 = beta52-alpha52; gamma53 = beta53-alpha53; gamma54 = beta54-alpha54
        gamma61 = beta61-alpha61; gamma62 = beta62-alpha62; gamma63 = beta63-alpha63; gamma64 = beta64-alpha64; gamma65 = beta65-alpha65
        gamma71 = beta71-alpha71; gamma72 = beta72-alpha72; gamma73 = beta73-alpha73; gamma74 = beta74-alpha74; gamma75 = beta75-alpha75; gamma76 = beta76-alpha76
        gamma81 = beta81-alpha81; gamma82 = beta82-alpha82; gamma83 = beta83-alpha83; gamma84 = beta84-alpha84; gamma85 = beta85-alpha85; gamma86 = beta86-alpha86; gamma87 = beta87-alpha87

        c1 = beta81; c2 = beta82; c3 = beta83; c4 = beta84; c5 = beta85; c6 = beta86; c7 = beta87; c8 = gamma
        chat1 = beta71; chat2 = beta72; chat3 = beta73; chat4 = beta74; chat5 = beta75; chat6 = beta76; chat7 = gamma

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
        t3, y3 = t0, (y0**ω + alpha41 * k1**ω + alpha42 * k2**ω + alpha43 * k3**ω).ω
        f3 = terms.vf_prod(t3, y3, args, control)
        Jk3 = J.mv(k3)
        b4 = (f3**ω + gamma41 * Jk1**ω + gamma42 * Jk2**ω + gamma43 * Jk3**ω).ω
        k4 = jax.scipy.linalg.lu_solve( LU_and_piv, b4 )

        # Stage 5
        t4, y4 = t0, (y0**ω + alpha51 * k1**ω + alpha52 * k2**ω + alpha53 * k3**ω + alpha54 * k4**ω).ω
        f4 = terms.vf_prod(t4, y4, args, control)
        Jk4 = J.mv(k4)
        b5 = (f4**ω + gamma51 * Jk1**ω + gamma52 * Jk2**ω + gamma53 * Jk3**ω + gamma54 * Jk4**ω).ω
        k5 = jax.scipy.linalg.lu_solve( LU_and_piv, b5 )

        # Stage 6
        t5, y5 = t0, (y0**ω + alpha61 * k1**ω + alpha62 * k2**ω + alpha63 * k3**ω + alpha64 * k4**ω + alpha65 * k5**ω).ω
        f5 = terms.vf_prod(t5, y5, args, control)
        Jk5 = J.mv(k5)
        b6 = (f5**ω + gamma61 * Jk1**ω + gamma62 * Jk2**ω + gamma63 * Jk3**ω + gamma64 * Jk4**ω + gamma65 * Jk5**ω).ω
        k6 = jax.scipy.linalg.lu_solve( LU_and_piv, b6 )

        # Stage 7
        t6, y6 = t0, (y0**ω + alpha71 * k1**ω + alpha72 * k2**ω + alpha73 * k3**ω + alpha74 * k4**ω + alpha75 * k5**ω + alpha76 * k6**ω).ω
        f6 = terms.vf_prod(t6, y6, args, control)
        Jk6 = J.mv(k6)
        b7 = (f6**ω + gamma71 * Jk1**ω + gamma72 * Jk2**ω + gamma73 * Jk3**ω + gamma74 * Jk4**ω + gamma75 * Jk5**ω + gamma76 * Jk6**ω).ω
        k7 = jax.scipy.linalg.lu_solve( LU_and_piv, b7 )

        # Stage 8
        t7, y7 = t0, (y0**ω + alpha81 * k1**ω + alpha82 * k2**ω + alpha83 * k3**ω + alpha84 * k4**ω + alpha85 * k5**ω + alpha86 * k6**ω +  + alpha87 * k7**ω).ω
        f7 = terms.vf_prod(t7, y7, args, control)
        Jk7 = J.mv(k7)
        b8 = (f7**ω + gamma81 * Jk1**ω + gamma82 * Jk2**ω + gamma83 * Jk3**ω + gamma84 * Jk4**ω + gamma85 * Jk5**ω + gamma86 * Jk6**ω + gamma87 * Jk7**ω).ω
        k8 = jax.scipy.linalg.lu_solve( LU_and_piv, b8 )
        
         
        # Advance Solution
        y1 = (y0**ω + c1 * k1**ω + c2 * k2**ω + c3 * k3**ω + c4 * k4**ω + c5 * k5**ω + c6 * k6**ω + c7 * k7**ω + c8 * k8**ω).ω
        y_error = ((c6-chat6)*k6**ω + (c7-chat7)*k7**ω + c8*k8**ω).ω

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