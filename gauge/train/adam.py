import jax
import numpy as np
import optax
from jax import jit
from tqdm.auto import tqdm

str_to_opt = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "amsgrad": optax.amsgrad,
    "adabelief": optax.adabelief,
    "sgd": optax.sgd,
    "lbfgs": optax.lbfgs,
    "muon": optax.contrib.muon
}


def adam_opt(
    theta_init,
    loss_fn,
    args_fn,
    init_state=None,
    steps=1000,
    learning_rate=1e-3,
    scheduler=True,
    warm_up=False,
    verbose=False,
    loss_tol=None,
    optimizer="adam",
    grad_clip_norm: float | None = None,
    key=None,
    loss_key=False,
    save_history=False,
    n_save=1000,
    has_aux=False,
):
    # If scheduler is used, optionally add a 4k-step warmup + cosine decay
    if scheduler:
        if warm_up:
            warmup_steps = min(int(steps * 0.05), 5000)
            decay_steps = max(0, steps - warmup_steps)

            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=learning_rate * 5e-4  # final multiplicative factor
            )
            learning_rate = schedule
        else:
            # Otherwise, just use standard cosine decay
            learning_rate = optax.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=steps, alpha=5e-4
            )

    opti_f = str_to_opt[optimizer]
    optimizer = opti_f(learning_rate=learning_rate)
    if grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(grad_clip_norm)),
            optimizer,
        )

    state = optimizer.init(theta_init)
    if init_state is not None:
        state = init_state

    @jit
    def step(params, state, args, lkey):
        if loss_key:
            loss_out, grads = jax.value_and_grad(
                loss_fn, has_aux=has_aux)(params, *args, lkey)
        else:
            loss_out, grads = jax.value_and_grad(
                loss_fn, has_aux=has_aux)(params, *args)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return loss_out, params, state

    params = theta_init
    pbar = tqdm(range(steps), disable=not verbose, colour="blue")
    loss_history = []
    param_history = []

    for i in pbar:
        if callable(args_fn):
            if key is not None:
                key, skey = jax.random.split(key)
                args = args_fn(skey)
            else:
                args = args_fn()
        else:
            args = args_fn

        key, lkey = jax.random.split(key)
        loss_out, params_new, state_new = step(params, state, args, lkey)

        if has_aux:
            loss_value, aux = loss_out
            FIELD_WIDTH = 12
            segments = [f"loss: {loss_value:10.6f}".ljust(FIELD_WIDTH)]
            for k, v in aux.items():
                segments.append(f"{k}: {v:10.6f}".ljust(FIELD_WIDTH))
            pbar.set_description(" | ".join(segments), refresh=False)
        else:
            loss_value = loss_out
            pbar.set_description(f"loss: {loss_value:.6f}", refresh=False)

        if i % n_save == 0:
            loss_history.append(loss_value)
            if save_history:
                param_history.append(params)

        params = params_new
        state = state_new

        if loss_tol is not None and loss_value < loss_tol:
            break

    loss_history = np.asarray(loss_history)

    if save_history:
        return params, loss_history, param_history

    return params, loss_history
