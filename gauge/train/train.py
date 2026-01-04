
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


def train_model(cfg: Config, dataloader, loss, params_init, key_opt, has_aux=False, name=''):
    opt_cfg = cfg.optimizer
    opt_params, ema_params, loss_history = run_train(
        params_init,
        dataloader,
        loss,
        opt_cfg.iters,
        optimizer=opt_cfg.optimizer,
        learning_rate=opt_cfg.lr,
        scheduler=opt_cfg.scheduler,
        rng=key_opt,
        has_aux=has_aux,
        ema_decay=opt_cfg.ema_decay or None,
        grad_clip_norm=opt_cfg.grad_clip or None
    )

    R.RESULT[f"{name}_opt_params"] = opt_params
    R.RESULT[f"{name}_ema_params"] = ema_params
    R.RESULT[f"{name}_loss_history"] = loss_history
    if len(loss_history) > 0:
        R.RESULT[f"{name}_final_loss"] = loss_history[-1]

    # best practice: return EMA for sampling/eval (falls back to opt_params if EMA disabled)
    return ema_params


# ---------- multi-device helpers ----------
_N_DEVICES = jax.local_device_count()
_MULTI_DEVICE = _N_DEVICES > 1


def run_train(
    params_init,
    dataloader,
    fwd_fn,
    iters,
    optimizer: str = 'adamw',
    learning_rate: float = 1e-3,
    scheduler: str = 'cos',
    N: int = 2048,
    rng=None,
    has_aux=False,
    ema_decay: float | None = None,
    grad_clip_norm: float | None = None,
):
    if rng is None:
        rng = jax.random.PRNGKey(1)

    if scheduler is not None:
        if scheduler == 'cos':
            learning_rate = optax.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=iters, alpha=1e-3
            )
        elif scheduler == 'const':
            learning_rate = optax.constant_schedule(value=learning_rate)
        elif scheduler == 'linear':
            learning_rate = optax.linear_schedule(
                init_value=learning_rate,
                end_value=learning_rate * 1e-3,
                transition_steps=iters
            )
        elif scheduler == 'warmup':
            learning_rate = optax.warmup_constant_schedule(
                init_value=1e-6, peak_value=learning_rate, warmup_steps=10_000
            )

    opti_f = str_to_opt[optimizer]

    # NEW: optionally clip gradients before the optimizer transform
    base_tx = opti_f(learning_rate=learning_rate)
    if grad_clip_norm is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(float(grad_clip_norm)),
            base_tx,
        )
    else:
        tx = base_tx

    opt_state = tx.init(params_init)

    use_ema = ema_decay is not None
    ema_step_size = 0.0 if not use_ema else (1.0 - float(ema_decay))

    @jax.jit
    def train_step(params, ema_params, rng, opt_state, x, cls):
        def _loss_fn(p):
            return fwd_fn(p, x, cls, rng)

        loss_out, grads = jax.value_and_grad(_loss_fn, has_aux=has_aux)(params)

        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        if use_ema:
            new_ema_params = optax.incremental_update(
                new_params, ema_params, step_size=ema_step_size)
        else:
            new_ema_params = new_params

        return new_params, new_ema_params, new_opt_state, loss_out, rng

    loss_history = []
    interval = max(1, iters // N)

    opt_params = params_init
    ema_params = params_init  # if EMA disabled, we’ll just keep ema_params == opt_params

    pbar = tqdm(range(iters), colour='blue')
    dl_iter = iter(dataloader)

    if _MULTI_DEVICE:
        mesh = jax.make_mesh((_N_DEVICES,), ('batch',))
        SHARD_BATCH = NamedSharding(mesh, P('batch'))
        SHARD_REPL = NamedSharding(mesh, P())
    else:
        mesh = SHARD_BATCH = SHARD_REPL = None

    def jax_put(x, sharding=None, float_16=False):
        if float_16:
            x = x.astype(jnp.bfloat16)
        return jax.device_put(x, sharding) if sharding is not None else jax.device_put(x)

    opt_params = jax_put(opt_params, SHARD_REPL)
    ema_params = jax_put(ema_params, SHARD_REPL)
    opt_state = jax_put(opt_state,  SHARD_REPL)

    for step in pbar:
        x, cls = next(dl_iter)

        x = jax_put(x, SHARD_BATCH)
        cls = jax_put(cls, SHARD_BATCH) if cls is not None else None

        rng, skey = jax.random.split(rng)

        opt_params, ema_params, opt_state, loss_out, rng = train_step(
            opt_params, ema_params, skey, opt_state, x, cls
        )

        if has_aux:
            loss_value, aux = loss_out
            FIELD_WIDTH = 12
            segments = [f"loss: {loss_value:10.4f}".ljust(FIELD_WIDTH)]
            for k, v in aux.items():
                segments.append(f"{k}: {v:10.4f}".ljust(FIELD_WIDTH))
            pbar.set_description(" | ".join(segments), refresh=False)
        else:
            loss_value = loss_out
            pbar.set_description(f"loss: {loss_value:.6f}", refresh=False)

        if (step % interval) == 0:
            loss_history.append(loss_value)

    loss_history = np.array(loss_history, dtype=np.float32)
    return opt_params, ema_params, loss_history
