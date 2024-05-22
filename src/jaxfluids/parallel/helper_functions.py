import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.multihost_utils import sync_global_devices

def synchronize_jf(is_multihost: bool = False) -> None:
    sync_buffer = jnp.arange(jax.local_device_count())
    sync_buffer = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(sync_buffer)
    sync_buffer.block_until_ready()
    if is_multihost:
        sync_global_devices("complete")
