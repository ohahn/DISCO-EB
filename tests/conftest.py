import json

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


CLASS_DATA = json.load(open("tests/resources/CLASS_data.json"))

RECFAST_DISCO_EB_DATA = json.load(open("tests/resources/RECFAST_DISCO_EB_data.json"))

k_CLASS = jnp.array(CLASS_DATA["k"])
Pkbc_CLASS = jnp.array(CLASS_DATA["Pkbc"])

a_RECFAST = jnp.array(RECFAST_DISCO_EB_DATA["a"])
xe_RECFAST = jnp.array(RECFAST_DISCO_EB_DATA["xe"])