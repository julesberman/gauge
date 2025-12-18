import glob
import inspect
import os
import random
import string
from functools import partial
from time import time
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap
from tqdm.auto import tqdm


def load_with_pattern(directory, filename_pattern):
    search_pattern = os.path.join(directory, filename_pattern)
    matching_files = glob.glob(search_pattern)
    return matching_files


def randkey():
    return jax.random.PRNGKey(random.randint(-1e12, 1e12))


def randkeys(num):
    k = jax.random.PRNGKey(random.randint(-1e12, 1e12))
    return jax.random.split(k, num=num)


def unique_id(n) -> str:
    """creates unique alphanumeric id w/ low collision probability"""
    chars = string.ascii_letters + string.digits  # 64 choices
    id_str = "".join(random.choice(chars) for _ in range(n))
    return id_str


def epoch_time(decimals=0) -> int:
    return int(time() * (10 ** (decimals)))


def pts_array_from_space(space):
    m_grids = jnp.meshgrid(*space, indexing="ij")
    x_pts = jnp.asarray([m.flatten() for m in m_grids]).T
    return x_pts


def pshape(*args, title=''):
    # Get the previous frame in the stack (i.e., the caller's frame)
    frame = inspect.currentframe().f_back
    # Get the caller's local variables
    local_vars = frame.f_locals

    # Build a mapping from id(value) to name(s)
    value_to_names = {}
    for var_name, value in local_vars.items():
        value_id = id(value)
        if value_id in value_to_names:
            value_to_names[value_id].append(var_name)
        else:
            value_to_names[value_id] = [var_name]

    dlim = " | "
    if title:
        print(title, end=' || ')

    for arg in args:
        value_id = id(arg)
        var_names = value_to_names.get(value_id, ["unknown"])
        # Join multiple variable names if they reference the same object
        var_name_str = ", ".join(var_names)
        if hasattr(arg, "shape"):
            print(f"{var_name_str}: {arg.shape}", end=dlim)
        else:
            print(f"{var_name_str}: no_shape", end=dlim)
    print()


def print_stats(x):
    print(
        f"shape: {x.shape} min: {x.min():.5f}, max: {x.max():.5f}, mean: {x.mean():.5f}, std: {x.std():.5f}, dtype: {x.dtype}, type: {type(x)}")


def get_rand_idx(key, N, bs):
    if bs > N:
        bs = N
    idx = jnp.arange(0, N)
    return jax.random.choice(key, idx, shape=(bs,), replace=False)


def meanvmap(f, in_axes=(0), mean_axes=(0,)):
    return lambda *fargs, **fkwargs: jnp.mean(
        vmap(f, in_axes=in_axes)(*fargs, **fkwargs), axis=mean_axes
    )


def tracewrap(f, axis1=0, axis2=1):
    return lambda *fargs, **fkwargs: jnp.trace(
        f(*fargs, **fkwargs), axis1=axis1, axis2=axis2
    )


def normwrap(f, axis=None, flatten=False):
    if flatten:
        return lambda *fargs, **fkwargs: jnp.linalg.norm(f(*fargs, **fkwargs).reshape(-1))
    else:
        return lambda *fargs, **fkwargs: jnp.linalg.norm(f(*fargs, **fkwargs), axis=axis)


def batchvmap(f, batch_size, in_arg=0, batch_dim=0, pbar=False):

    def wrap(*fargs, **fkwarg):
        fargs = list(fargs)
        X = fargs[in_arg]
        n_batches = jnp.ceil(X.shape[batch_dim] // batch_size).astype(int)
        n_batches = max(1, n_batches)
        batches = jnp.array_split(X, n_batches, axis=batch_dim)

        in_axes = [None] * len(fargs)
        in_axes[in_arg] = batch_dim
        v_f = vmap(f, in_axes=in_axes)
        result = []
        if pbar:
            batches = tqdm(batches)
        for B in batches:
            fargs[in_arg] = B
            a = v_f(*fargs, **fkwarg)
            result.append(a)

        return jnp.concatenate(result)

    return wrap


def sqwrap(f):
    return lambda *fargs, **fkwargs: jnp.squeeze(f(*fargs, **fkwargs))


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def normalize(
    x: jnp.ndarray,
    method: str = "zscore",
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    eps: float = 1e-8,
    keepdims: bool = False,
    stats: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Normalize `x` using different conventions.

    Parameters
    ----------
    x : Array
        Input array.
    method : {"zscore", "std", "minmax", "01", "sym", "-11"}, default "zscore"
        - "zscore" / "std": (x - mean) / std
        - "minmax" / "01":  map min->0, max->1
        - "sym" / "-11":    map min->-1, max->1
    axis : int or tuple of int, optional
        Axes over which to compute statistics. If None, use all elements.
    eps : float, default 1e-8
        Small value to avoid division by zero when scale is very small.
    keepdims : bool, default False
        Whether to keep reduced dimensions when computing statistics.
        This can make broadcasting more explicit.
    stats : (shift, scale), optional
        If given, reuse these stats instead of recomputing them.
        Useful for applying training-set normalization to test data.

    Returns
    -------
    x_norm : Array
        Normalized array.
    (shift, scale) : tuple of Array
        The shift and scale used so you can reuse them later.

    Notes
    -----
    To invert the normalization:
        x = x_norm * scale + shift
    """
    method = method.lower()

    if stats is not None:
        shift, scale = stats
    else:
        if method in {"minmax", "01"}:
            x_min = jnp.min(x, axis=axis, keepdims=keepdims)
            x_max = jnp.max(x, axis=axis, keepdims=keepdims)
            shift = x_min
            scale = x_max - x_min

        elif method in {"sym", "-11"}:
            x_min = jnp.min(x, axis=axis, keepdims=keepdims)
            x_max = jnp.max(x, axis=axis, keepdims=keepdims)
            shift = (x_max + x_min) / 2.0
            scale = (x_max - x_min) / 2.0

        elif method in {"zscore", "std"}:
            shift = jnp.mean(x, axis=axis, keepdims=keepdims)
            scale = jnp.std(x, axis=axis, keepdims=keepdims)

        else:
            raise ValueError(f"Unknown normalization method: {method!r}")

    # Avoid division by zero / extremely small scales
    scale = jnp.where(jnp.abs(scale) < eps, 1.0, scale)

    x_norm = (x - shift) / scale
    return x_norm, (shift, scale)


def fold_in_data(*args):
    s = 0.0
    for a in args:
        s += jnp.cos(jnp.linalg.norm(a))
    s *= 1e6
    s = s.astype(jnp.int32)
    return s


def combine_keys(df, n_k, k_arr):
    df[n_k] = df[k_arr].agg(lambda x: "~".join(x.astype(str)), axis=1)
    return df


def jacrand(rng_key, fun, argnums=0):
    """
    Like jacrev, but instead of returning a full Jacobian, it returns a
    single Jacobian-vector product in a random direction (of norm 1).

    Args:
        fun:      Function whose output we care about. Signature:
                  fun(*args, **kwargs) -> output pytree
        argnums:  int or tuple of ints, specifying which positional args
                  to differentiate against. (Default = 0)
        rng_key:  Optional PRNGKey to control randomness. If None,
                  defaults to jax.random.PRNGKey(0). Internally stored and
                  split each time you call the transformed function.

    Returns:
        A new function with the same signature as `fun` that returns:
            ( fun(*args, **kwargs), jvp_value )
        where jvp_value is the result of J_fun(*args, **kwargs) @ r_unit,
        and r_unit is a new random direction (norm 1) each call.
    """

    # Convert argnums to a tuple for uniform handling
    if isinstance(argnums, int):
        argnums_tuple = (argnums,)
    else:
        argnums_tuple = tuple(argnums)

    def wrapper(*args, **kwargs):
        # Extract the "differentiable" arguments (the ones in argnums)
        # as the 'primals' for jvp
        primals = tuple(args[i] for i in argnums_tuple)

        # Build a partial version of fun that only expects the
        # differentiable arguments, while capturing the rest by closure.
        def partial_fun(*dyn_args):
            # Rebuild the entire argument list
            full_args = list(args)
            for idx, val in zip(argnums_tuple, dyn_args, strict=False):
                full_args[idx] = val
            # Call original function
            return fun(*full_args, **kwargs)

        subkeys = jax.random.split(rng_key, len(argnums_tuple))

        tangent_subset = []
        for p, sk in zip(primals, subkeys, strict=False):
            r = jax.random.normal(sk, shape=p.shape)
            # Protect against zero-norm corner cases
            norm = jnp.linalg.norm(r)
            r_unit = r / (norm + 1e-12)
            tangent_subset.append(r_unit)

        tangent_subset = tuple(tangent_subset)

        # Get (primal_out, jvp_out)
        y, jvp = jax.jvp(partial_fun, primals, tangent_subset)

        jvp = jnp.linalg.norm(jvp.reshape(-1))

        return jvp

    return wrapper


def get_cpu_count() -> int:
    cpu_count = None
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_count = len(os.sched_getaffinity(0))
            return cpu_count
        except Exception:
            pass

    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count

    try:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
    except Exception:
        pass

    print("could not get cpu count, returning 1")

    return 1


def get_available_ram_gb():
    paths = [
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",            # older cgroup v1
        "/sys/fs/cgroup/memory.max",                              # cgroup v2
    ]

    for path in paths:
        try:
            with open(path, "r") as f:
                limit = int(f.read().strip())
                if limit > 0 and limit < 1 << 60:  # filter out "no limit" sentinel values
                    return limit / (1024**3)
        except FileNotFoundError:
            continue
    return -1


# def hutch_div(f, n_samples=1, rv_type="rademacher", argnum=0):
#     """
#     Hutchinson-style divergence estimator for a vector field f.

#     Parameters
#     ----------
#     f : callable
#         Function f(*args) with respect to which we want div wrt args[argnum].
#         Assumes: f(..., x, ...) has same shape as x at position argnum.
#     n_samples : int
#         Number of Hutchinson probe vectors per call.
#     rv_type : str
#         "gaussian", "rademacher", "sphere"
#     argnum : int
#         Index of argument to take divergence with respect to.

#     Returns
#     -------
#     div : callable
#         Function div(*args, key) -> scalar divergence estimate.
#     """

#     def sample_vector(key, shape, dtype=jnp.float32):
#         if rv_type == "gaussian":
#             return jax.random.normal(key, shape, dtype=dtype)
#         elif rv_type == "rademacher":
#             return jax.random.rademacher(key, shape, dtype=dtype)
#         elif rv_type == "sphere":
#             v = jax.random.normal(key, shape, dtype=dtype)
#             return v / (jnp.linalg.norm(v) + 1e-7)
#         else:
#             raise ValueError(f"Unknown rv_type {rv_type!r}")

#     @jax.jit
#     def div(key, *args):
#         x = args[argnum]

#         def g(x_arg):
#             # Reconstruct argument list with x_arg in position argnum
#             new_args = list(args)
#             new_args[argnum] = x_arg
#             return f(*new_args)

#         keys = jax.random.split(key, n_samples)

#         def one(k):
#             v = sample_vector(k, x.shape, dtype=x.dtype)
#             _, jvp_val = jax.jvp(g, (x,), (v,))
#             return jnp.vdot(v, jvp_val)

#         return jax.vmap(one)(keys).mean()

#     return div


def hutch_div(f, argnum: int = 1, n_samples: int = 1):
    """
    Hutchinson divergence estimator for:
      - f: R^d -> R^d           (x.shape = (d,))
      - f: R^{b,d} -> R^{b,d}   (x.shape = (b,d))

    Returns div(*args, key, return_fwd=False) whose output is:
      - div_est                    if return_fwd=False
      - (div_est, f_x)             if return_fwd=True

    Shapes:
      - div_est: () or (b,)
      - f_x:     (d,) or (b, d)
    """

    @partial(jax.jit, static_argnames=("return_fwd",))
    def div(*args, key, return_fwd: bool = False):
        x = args[argnum]

        def g(x_arg):
            new_args = list(args)
            new_args[argnum] = x_arg
            return f(*new_args)

        keys = jax.random.split(key, n_samples)

        # Compute primal once, and get a cached linear map v ↦ J(x)v
        f_x, lin = jax.linearize(g, x)

        def one(k):
            v = jax.random.rademacher(k, shape=x.shape, dtype=x.dtype)
            jvp_val = lin(v)  # = J(x) v
            return jnp.sum(v * jvp_val, axis=-1)  # () or (b,)

        div_est = jax.vmap(one)(keys).mean(axis=0)

        return (div_est, f_x) if return_fwd else div_est

    return div
