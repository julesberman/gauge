

import flax.linen as nn
import jax.numpy as jnp


class DNN(nn.Module):
    width: int
    depth: int
    out_features: int
    activate_last: bool = False
    residual: bool = False
    n_classes: int = 10

    @nn.compact
    def __call__(self,  x, time, class_l):
        A = nn.gelu
        if time is not None:
            time = nn.Dense(
                self.width,
                use_bias=self.use_bias,
            )(time)
            time = A(time)

        if class_l is not None:
            class_l = nn.Embed(self.n_classes, self.width)(class_l)

        temb = None
        if time is not None and class_l is not None:
            temb = jnp.concatenate(
                [time, class_l], axis=-1)
        if time is None and class_l is not None:
            temb = class_l
        if time is not None and class_l is None:
            temb = time

        if temb is not None:
            temb = MLP(
                width=self.width,
                depth=2,
                use_bias=self.use_bias,
            )(temb)

        last_x = None
        for _ in range(self.depth):
            D = nn.Dense(
                self.width,
                use_bias=self.use_bias,
            )

            last_x = x
            x = D(x)
            x = A(x)

            if temb is not None:
                bias = MLP(
                    width=self.width,
                    depth=2,
                )(temb)

                bias = bias.reshape(-1,  self.width)
                x = x + bias

            if self.residual and last_x is not None:
                if x.shape == last_x.shape:
                    x = x + last_x

        x = nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
        )(x)

        return x


class MLP(nn.Module):
    width: int
    depth: int
    out_features: int | None = None
    use_bias: bool = True
    activate_last: bool = False

    @nn.compact
    def __call__(self, x):
        A = nn.gelu
        out_features = self.out_features
        if out_features is None:
            out_features = self.width

        for _ in range(self.depth - 1):
            D = nn.Dense(
                self.width,
                use_bias=self.use_bias,
            )
            x = D(x)
            x = A(x)

        D = nn.Dense(
            out_features,
            use_bias=self.use_bias,
        )
        x = D(x)

        if self.activate_last:
            x = A(x)

        return x
