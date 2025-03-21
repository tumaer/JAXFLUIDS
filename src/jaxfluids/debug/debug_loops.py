def debug_while_loop(cond_fun, body_fun, init_val):  # pragma: no cover
    """
    for debugging purposes, use this instead of jax.lax.while_loop
    """
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val

def debug_fori_loop(start, stop, body_fun, init_val):  # pragma: no cover
    """
    for debugging purposes, use this instead of jax.lax.fori_loop
    """
    val = init_val
    for i in range(start, stop):
        val = body_fun(i, val)
    return val