
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optax.contrib
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm.auto import tqdm

import gauge.io.result as R
from gauge.config.config import Config

str_to_opt = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "lbfgs": optax.lbfgs,
    "muon": optax.contrib.muon
}


def train_model(cfg: Config, net, dataloader, score_loss, params_init, key_opt, name=''):

    opt_cfg = cfg.optimizer

    opt_params, loss_history = run_train(net, params_init, dataloader, score_loss, opt_cfg.iters, optimizer=opt_cfg.optimizer,
                                         learning_rate=opt_cfg.lr, scheduler=opt_cfg.scheduler, rng=key_opt)

    R.RESULT[f"{name}_opt_params"] = opt_params
    R.RESULT[f"{name}_loss_history"] = loss_history

    if len(loss_history) > 0:
        R.RESULT[f"{name}_final_loss"] = loss_history[-1]

    return opt_params


# ---------- multi-device helpers ----------
_N_DEVICES = jax.local_device_count()
_MULTI_DEVICE = _N_DEVICES > 1


def run_train(
    net,
    params_init,
    dataloader,
    fwd_fn,
    iters,
    optimizer: str = 'adamw',   # Default to adamw
    learning_rate: float = 1e-3,
    scheduler: str = 'cos',
    N: int = 2048,               # Record the training loss 1000 times by default
    rng=None,                    # Accept a JAX key instead of an int seed
):
    """
    Train a Flax module with dropout using Optax optimizers.

    Args:
        net: A Flax module with an .apply() method. Typically called as
             net.apply(params, x, l, rngs={'dropout': dropout_rng}, train=True).
        params_init: The initial parameters of the network.
        dataloader: An iterator returning (x, y, l) batches.
        fwd_fn: A function taking (**) -> scalar loss.
        iters: Number of training iterations (gradient steps).
        optimizer: Which optimizer to use (str). E.g., 'adam', 'adamw' (default).
        learning_rate: Base learning rate (float).
        scheduler: If True, use a piecewise schedule (warm-up + constant).
        N: Number of times to record the training loss over the course of training.
        rng: A JAX PRNGKey for dropout and random operations.
        val_fn: (Optional) A callable that takes (params) -> scalar validation metric.
        val_steps: Evaluate `val_fn` every `val_steps` iterations, but only if
                   `val_fn` is not None and `val_steps >= iters`.

    Returns:
        opt_params: The trained parameters after `iters` iterations (final).
        val_params:
            - If validation is performed (val_fn not None and val_steps >= iters),
              this will be the parameters that achieved the *lowest* validation scalar.
            - Otherwise, just the same as opt_params.
        loss_history: A numpy array of recorded loss values (length ~ N).
    """
    # ---------------------
    # Setup default RNG key
    # ---------------------
    if rng is None:
        rng = jax.random.PRNGKey(1)

    if scheduler is not None:
        # Otherwise, just use standard cosine decay
        if scheduler == 'cos':
            learning_rate = optax.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=iters, alpha=1e-3
            )
        elif scheduler == 'const':
            learning_rate = optax.constant_schedule(value=learning_rate)
        elif scheduler == 'linear':
            learning_rate = optax.linear_schedule(
                init_value=learning_rate, end_value=learning_rate*1e-3, transition_steps=iters)
        elif scheduler == 'warmup':
            learning_rate = optax.warmup_constant_schedule(
                init_value=1e-6, peak_value=learning_rate, warmup_steps=10_000)
        else:
            learning_rate = learning_rate

    opti_f = str_to_opt[optimizer]
    tx = opti_f(learning_rate=learning_rate)

    # -------------------------
    # 3. Initialize the OptState
    # -------------------------
    opt_state = tx.init(params_init)

    # --------------------
    # 4. Define train_step
    # --------------------

    @jax.jit
    def train_step(params, rng, opt_state, x, cls):
        """
        Single step of forward, loss computation, backward, and optimizer update.
        """

        def _loss_fn(p):
            return fwd_fn(p, x, cls, rng)

        # Compute gradients
        loss_value, grads = jax.value_and_grad(_loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss_value, rng

    loss_history = []
    interval = max(1, iters // N)

    opt_params = params_init
    pbar = tqdm(range(iters), colour='blue')

    dl_iter = iter(dataloader)
    vkey, key = jax.random.split(rng)

    def jax_put(x, replicated=False, float_16=False):

        if float_16:
            x = x.astype(jnp.float16)
        if _MULTI_DEVICE:
            mesh = jax.make_mesh((_N_DEVICES,), ('batch',))
            if replicated:
                sharding = NamedSharding(mesh, P())
            else:
                sharding = NamedSharding(mesh, P('batch'))
            x = jax.device_put(x, sharding)
        else:
            x = jax.device_put(x)
        return x

    opt_params = jax_put(opt_params, replicated=True)
    opt_state = jax_put(opt_state, replicated=True)

    for step in pbar:
        x, cls = next(dl_iter)  # get batch

        if cls is None:
            x = jax_put(x, float_16=False)
        else:
            x, cls = [jax_put(bb, float_16=False) for bb in (x, cls)]

        rng, skey = jax.random.split(rng)
        # Take a single training step
        opt_params, opt_state, loss_value, rng = train_step(
            opt_params, skey, opt_state, x, cls
        )

        # Update the progress bar
        pbar.set_description(f"loss: {loss_value:.6f}")

        # Record the loss every `interval` steps
        if (step % interval) == 0:
            loss_history.append(loss_value)

    loss_history = np.array(loss_history, dtype=np.float32)

    return opt_params, loss_history
