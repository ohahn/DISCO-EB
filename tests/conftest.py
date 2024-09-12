import json

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


CLASS_DATA = json.load(open("tests/resources/CLASS_data.json"))

k_CLASS = jnp.array(CLASS_DATA["k"])
Pkbc_CLASS = jnp.array(CLASS_DATA["Pkbc"])

