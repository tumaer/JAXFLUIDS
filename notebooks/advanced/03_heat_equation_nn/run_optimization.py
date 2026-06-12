from collections.abc import Callable
from pathlib import Path
import time
from typing import Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import optax

from jaxfluids import InputManager, SimulationManager
from jaxfluids.data_types.ml_buffers import (
    ParametersSetup,
    CallablesSetup,
    DiffusiveFluxesSetup,
)
from jaxfluids.feed_forward.data_types import FeedForwardSetup

from flax import linen as nn

Array = jax.Array

FEED_FORWARD_OUTER_STEPS = 200
FEED_FORWARD_INNER_STEPS = 100
PHYSICAL_TIMESTEP_SIZE = 2.5e-5
TOTAL_STEPS = 300
OUTPUT_DIR = "pngs"


class MLP(nn.Module):
    @nn.compact
    def __call__(self, T: Array) -> Array:
        x = T[..., None]
        x = nn.Dense(32, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        x = nn.gelu(x)
        x = nn.Dense(32, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        x = nn.gelu(x)
        x = nn.Dense(32, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        x = nn.gelu(x)
        x = nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        thermal_cond = 1.0 + nn.softplus(x)
        return thermal_cond[..., 0]


def gaussian_thermal_conductivity(T: Array) -> Array:
    return 1.0 + 2.0 * jnp.exp(-0.5 * ((T - 1.5) / 0.15)**2)


def build_feed_forward(
        sim_manager: SimulationManager,
        model: MLP | None = None,
    ) -> Callable[[Any | None], Array]:

    domain_information = sim_manager.domain_information
    material_manager = sim_manager.material_manager

    feed_forward_setup = FeedForwardSetup(
        outer_steps=FEED_FORWARD_OUTER_STEPS,
        inner_steps=FEED_FORWARD_INNER_STEPS,
        is_include_halos=False,
    )

    def initial_condition() -> Array:
        # Compute initial conditions
        primitives = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0]).reshape(5, 1, 1, 1)
        X, _ = domain_information.compute_device_mesh_grid()
        return jnp.broadcast_to(primitives, (5,) + X.shape)

    def extract_temperature(solution_array: Array) -> Array:
        # Extract temperature from the feed forward solution
        primitives = solution_array["primitives"][-1]
        temperature = material_manager.get_temperature(primitives)
        return jnp.squeeze(temperature)

    def thermal_cond(T: Array, *args: Any, params: Any, **kwargs: Any) -> Array:
        return model.apply(params, T)


    if model is None:
        ml_callables = CallablesSetup()
    else:
        ml_callables = CallablesSetup(
            diffusive_fluxes=DiffusiveFluxesSetup(
                thermal_conductivity=thermal_cond,
            )
        )

    def feed_forward(params: Any | None) -> Array:
        primitives = initial_condition()

        if params is None:
            ml_parameters = ParametersSetup()
        else:
            ml_parameters = ParametersSetup(
                diffusive_fluxes=DiffusiveFluxesSetup(
                    thermal_conductivity=params
                )
            )

        solution_array, _ = sim_manager._feed_forward(
            primes_init=primitives,
            physical_timestep_size=PHYSICAL_TIMESTEP_SIZE,
            t_start=0.0,
            feed_forward_setup=feed_forward_setup,
            ml_parameters=ml_parameters,
            ml_callables=ml_callables,
        )

        return extract_temperature(solution_array)

    return feed_forward


def build_loss_fn(
        feed_forward: Callable[[Any], Array],
    ) -> Callable[[Any, Array], tuple[Array, Array]]:

    def loss_fn(params: Any, temperature_ref: Array) -> tuple[Array, Array]:
        temperature = feed_forward(params)
        loss = jnp.mean(jnp.square(temperature - temperature_ref))
        return loss, temperature

    return loss_fn


def create_plot(
        meshgrid: tuple[Array, Array],
        temperature: Array,
        temperature_ref: Array,
        thermal_cond: Array,
        thermal_cond_ref: Array,
        loss: Array | float,
        step: int,
        save_path: Path,
    ) -> None:
    cmap = "Spectral_r"

    X, Y = meshgrid

    T_min, T_max = np.min(temperature_ref), np.max(temperature_ref)
    l_min, l_max = np.min(thermal_cond_ref), np.max(thermal_cond_ref)

    quants = (
        (r"$T$", temperature, T_min, T_max),
        (r"$T_{ref}$", temperature_ref, T_min, T_max),
        (r"$|T - T_{ref}|$", np.maximum(np.abs(temperature - temperature_ref), 1e-12), None, None),
        (r"$\lambda$", thermal_cond, l_min, l_max),
        (r"$\lambda_{ref}$", thermal_cond_ref, l_min, l_max),
        (r"$|\lambda - \lambda_{ref}|$", np.maximum(np.abs(thermal_cond - thermal_cond_ref), 1e-12), None, None),
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 7))
    ax = ax.flatten()

    pc = []
    for ii, (axi, (title, quant, vmin, vmax)) in enumerate(zip(ax, quants)):
        if ii == 2:
            norm = LogNorm(vmin=1e-4, vmax=1e-1)
        elif ii == 5:
            norm = LogNorm(vmin=1e-3, vmax=1e0)
        else:
            norm = None

        pci = axi.pcolormesh(
            X, Y, quant,
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            norm=norm,
        )
        pc.append(pci)

        axi.set_title(title)
        axi.set_box_aspect(1.0)
        axi.set_xticks([0, 0.5, 1.0])
        axi.set_yticks([0, 0.5, 1.0])

    cbar_ticks = (
        (1.0, 1.5, 2.0),
        (1.0, 1.5, 2.0),
        (1e-4, 1e-3, 1e-2, 1e-1),
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
        (1e-3, 1e-2, 1e-1, 1e0),
    )
    for axi, pci, ticks in zip(ax, pc, cbar_ticks):
        divider = make_axes_locatable(axi)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(pci, cax=cax, orientation="vertical")
        cbar.set_ticks(ticks)

    fname = rf"Step = {step:03d}, Loss = {loss:3.2e}"
    fig.suptitle(fname)
    plt.savefig(save_path / f"image_iter_{step:04d}.png", dpi=400, bbox_inches="tight")
    plt.close()


def main() -> None:
    base_path = Path(__file__).resolve().parent
    save_path = base_path / OUTPUT_DIR
    save_path.mkdir(exist_ok=True)

    input_manager = InputManager(
        str(base_path / "heat_equation.json"),
        str(base_path / "numerical_setup.json"),
    )
    sim_manager = SimulationManager(input_manager)

    domain_information = sim_manager.domain_information
    X, Y = domain_information.compute_mesh_grid()
    X, Y = X.squeeze(), Y.squeeze()

    model = MLP()
    params = model.init(jax.random.key(0), jnp.ones((1, 1)))

    feed_forward = build_feed_forward(sim_manager, None)

    temperature_ref = feed_forward(None)
    thermal_cond_ref = gaussian_thermal_conductivity(temperature_ref)
    thermal_cond = model.apply(params, temperature_ref)

    create_plot(
        (X, Y),
        temperature_ref,
        temperature_ref,
        thermal_cond,
        thermal_cond_ref,
        0.0, 0, save_path,
    )

    feed_forward = build_feed_forward(sim_manager, model)
    loss_fn = build_loss_fn(feed_forward)

    learning_rate = optax.exponential_decay(
        init_value=5e-3,
        transition_steps=40,
        decay_rate=0.8,
        staircase=True,
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def step_fn(params, temperature_ref, opt_state):
        (loss, temperature), grad = value_and_grad(params, temperature_ref)
        updates, opt_state = optimizer.update(grad, opt_state)

        thermal_cond = model.apply(params, temperature)

        updated_params = optax.apply_updates(params, updates)
        return loss, temperature, updated_params, opt_state, thermal_cond

    loss_history = []
    total_steps = TOTAL_STEPS
    steps_vec = np.arange(1, total_steps + 1)
    for i in steps_vec:
        t1 = time.time()
        (
            loss,
            temperature,
            updated_params,
            opt_state,
            thermal_cond
        ) = step_fn(params, temperature_ref, opt_state)
        loss.block_until_ready()
        t2 = time.time()

        wct = t2 - t1

        loss_history.append(loss)

        print(f"Step = {i:03d}, Loss = {loss:3.2e}, WCT = {wct:.2f}s", flush=True)

        create_plot(
            (X, Y),
            temperature,
            temperature_ref,
            thermal_cond,
            thermal_cond_ref,
            loss, i, save_path,
        )

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, i + 1), np.array(loss_history), color="black")
        ax.set_yscale("log")
        ax.set_xlabel("iter")
        ax.set_ylabel("loss")
        ax.set_xlim([0, i + 1])
        ax.set_ylim([1e-4, 1e-1])
        ax.set_box_aspect(1.0)
        plt.savefig(save_path / f"loss_history_iter_{i:04d}.png", dpi=400, bbox_inches="tight")
        plt.close()

        params = updated_params


if __name__ == "__main__":
    main()