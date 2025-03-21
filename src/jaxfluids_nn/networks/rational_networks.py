# Copyright 2024 The swirl_dynamics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Networks with Rational Activation functions.

References:
[1] Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend, Rational
    neural networks, arXiv preprint, arXiv:2004.01902 (2020).
"""

from typing import Any, Optional, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

Scalar = Any
Array = Any
ModuleDef = Any

def zero_cutoff(x: jnp.float64, epsilon: jnp.float64) -> jnp.float64:
    """Limits absolute value to always be larger than epsilon and preserves sign.

    Args:
        x: Input.
        epsilon: Small positive number to avoid the output being close to zero.

    Returns:
        x if abs(x) > epsilon, epsilon*sign(x) otherwise. It is assumed that the
        input x is a scalar.
    """
    sign_x = jax.lax.cond(x >= 0.0, lambda x: 1.0, lambda x: -1.0, x)
    return jax.lax.cond(
        jnp.abs(x) < epsilon,
        lambda x: epsilon * sign_x,
        lambda x: x,
        x,
    )


class RationalLayer(nn.Module):
    """Implementation of a trainable rational layer as described in [1].

    For each input xᵢ, this layers returns p(xᵢ)/q(xᵢ), where the polynomials
    are characterized by its coefficients, i.e.,
    p(x) = p_params[0] + p_params[0]*x + p_params[deg_pols[0]+1]*(x**deg_pols[0]),
    and
    q(x) = q_params[0] + q_params[0]*x + q_params[deg_pols[1]+1]*(x**deg_pols[1]).

    All the coefficients p_params and q_params are trainable.

    In addition, we follow the initialization in [1], in which for a rational
    function of degrees (3, 2) the weights are such that the activation function
    approximates the ReLU function.

    Attributes:
        deg_pols: a tuple of two integers indicating the type of layer, i.e., the
        degree of each of the polynomials to the used. If the rational function is
        p/q, then: deg_pols[0] = degree of p, and deg_pols[1] = degree of q. By
        default we use deg_pols = (3, 2).
        cutoff: Shift for the thresholding.
    """

    deg_pols: Tuple[int, int] = (3, 2)
    dtype: jnp.dtype = jnp.float32
    cutoff: jnp.float32 | jnp.float64 | None = None

    def setup(self):
        """Initializes the parameters for a type (3,2) rational activation."""

        if self.deg_pols[0] == 3 and self.deg_pols[1] == 2:
            # If the function is of order (3, 2) then we use the weights in [1].
            init_p = jax.nn.initializers.constant(
                jnp.array([1.1915, 1.5957, 0.5, 0.0218])
            )
            init_q = jax.nn.initializers.constant(jnp.array([2.383, 0.0, 1.0]))
        else:
            # Otherwise use a normal initialization.
            # There is *no* guarantee that the layer will not output a NaN value.
            init_p = jax.nn.initializers.normal()
            init_q = jax.nn.initializers.normal()

        # Initializing the coefficients of the polynomals.
        self.p_params = self.param(
            'p_coeffs', init_p, (self.deg_pols[0] + 1,), self.dtype
        )
        self.q_params = self.param(
            'q_coeffs', init_q, (self.deg_pols[1] + 1,), self.dtype
        )

    def __call__(self, inputs: Array) -> Array:
        """Application of the rational function to the input.

        Args:
        inputs: points in which the rational function will be evaluated.

        Returns:
        The rational function evaluated p(x)/q(x) at the inputs.
        """

        x = inputs.astype(self.dtype)

        # Evaluating the polynomials.
        p = jnp.polyval(self.p_params, x)
        q = jnp.polyval(self.q_params, x)

        if self.cutoff:
            q = jax.vmap(zero_cutoff, in_axes=(0, None))(q, self.cutoff)

        return p / q


class UnsharedRationalLayer(nn.Module):
    """Rational layer with learnable parameters per neuron.

    In contrast to the Rational layer in which the rational function is
    independent of the particualr neuron being activated, in this case each neuron
    has its own sets of parameter, i.e., the activation is given by pᵢ(xᵢ)/qᵢ(xᵢ).
    Only (3, 2)-type rational functions are supported.


    Attributes:
        dtype: the dtype of the computation (default: float32).
        cutoff: Shift for the thresholding.
    """

    dtype: jnp.dtype = jnp.float32
    cutoff: Optional[jnp.float32 | jnp.float64 | None] = None

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
        inputs: The array to be evaluated.

        Returns:
        The transformed input.
        """
        x_i = inputs.astype(self.dtype)

        # Initializating the parameters in the network following [1].
        p_params_vect = jnp.array([1.1915, 1.5957, 0.5, 0.0218]).reshape((1, -1))
        q_params_vect = jnp.array([2.383, 0.0, 1.0]).reshape((1, -1))

        # Defining the initializers (required for trainable parameters).
        init_p = jax.nn.initializers.constant(
            p_params_vect * jnp.ones((x_i.shape[-1], 1))
        )
        init_q = jax.nn.initializers.constant(
            q_params_vect * jnp.ones((x_i.shape[-1], 1))
        )

        p_params = self.param('p_params', init_p, (x_i.shape[-1], 4), self.dtype)
        q_params = self.param('q_params', init_q, (x_i.shape[-1], 3), self.dtype)

        pol_fun = jax.vmap(jnp.polyval, in_axes=(0,-1), out_axes=(-1))

        p = pol_fun(p_params, x_i)
        q = pol_fun(q_params, x_i)

        if self.cutoff:
            q = jax.vmap(zero_cutoff, in_axes=(0, None))(q, self.cutoff)

        return p / q


class RationalMLP(nn.Module):
    """Simple multi-layer perceptron, with a rational activation function.

    Attributes:
        features: The sizes of the layers in the MLP.
        dtype: Type of the elements.
        multi_rational: boolean to use different polynomials for each neuron.
    """

    features: tuple[int, ...]
    dtype: Any = jnp.float32
    multi_rational: bool = False
    use_bias: bool = True
    # TODO: add precision flag to have more granular control

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = inputs.astype(self.dtype)

        for feat in self.features[:-1]:
            x = nn.Dense(feat, use_bias=self.use_bias, param_dtype=self.dtype)(x)
            if self.multi_rational:
                # Using a different activation function per neuron.
                x = UnsharedRationalLayer(dtype=self.dtype)(x)
            else:
                # Sharing the activation function among neurons.
                x = RationalLayer(dtype=self.dtype)(x)

        x = nn.Dense(
            self.features[-1], use_bias=self.use_bias, param_dtype=self.dtype
        )(x)

        return x