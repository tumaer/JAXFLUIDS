from collections.abc import Callable, Mapping
from pathlib import Path

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np
import optax

from jaxfluids import InputManager, SimulationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, DiffusiveFluxesSetup
from jaxfluids.feed_forward.data_types import FeedForwardSetup

Array = jax.Array


def build_feed_forward(sim_manager: SimulationManager) -> Callable[[Array], Array]:

    domain_information = sim_manager.domain_information

    # Advance enough steps for the channel flow to approach steady state.
    feed_forward_setup = FeedForwardSetup(
        outer_steps=200,
        inner_steps=100,
        is_include_halos=False,
    )

    def feed_forward(viscosity: Array) -> Array:
        primitives = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0]).reshape(5, 1, 1, 1)
        X, _ = domain_information.compute_device_mesh_grid()
        primitives = jnp.broadcast_to(primitives, (5,) + X.shape)

        # The optimizer controls viscosity through the differentiable ML buffer.
        ml_parameters = ParametersSetup(
            diffusive_fluxes=DiffusiveFluxesSetup(
                dynamic_viscosity=viscosity
            )
        )

        solution_array, _ = sim_manager._feed_forward(
            primes_init=primitives,
            physical_timestep_size=5e-5,
            t_start=0.0,
            feed_forward_setup=feed_forward_setup,
            ml_parameters=ml_parameters,
        )

        primitives = solution_array["primitives"][-1]
        return primitives[1, 0, :, 0]

    return feed_forward


def build_loss_fn(
        feed_forward: Callable[[Array], Array],
    ) -> Callable[[Array, Array], tuple[Array, Array]]:

    def loss_fn(viscosity: Array, velX_ref: Array) -> tuple[Array, Array]:
        velX = feed_forward(viscosity)
        loss = jnp.mean(jnp.square(velX - velX_ref))
        return loss, velX

    return loss_fn


def create_reference_plot(
        plot_data: Mapping[float, tuple[Array, Array]],
        save_path: Path,
    ) -> None:

    fig, ax = plt.subplots()

    for key, value in plot_data.items():
        y, velX = value
        nu = float(key)

        ax.plot(y, velX, color="red", linestyle="-")
        # Analytic Poiseuille profile for the same viscosity.
        ax.plot(y, 0.5 / nu * y * (1 - y), color="black", linestyle="--")

        # Label each curve near the channel center.
        y_label = 0.5
        u_label = np.interp(y_label, y, velX) + 0.005

        ax.text(
            y_label,
            u_label,
            rf"$\mu = {nu:.1f}$",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 0.275])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.set_box_aspect(1.0)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_optimization_plot(
        y: Array,
        velX: Array,
        velX_ref: Array,
        loss_history: list[Array],
        step: int,
        total_steps: int,
        viscosity: Array,
        loss: Array,
        save_path: Path,
    ) -> None:

    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    ax[0].plot(np.arange(1, step + 1), np.array(loss_history), color="black")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("iter")
    ax[0].set_ylabel("loss")
    ax[0].set_xlim([0, total_steps])
    ax[0].set_ylim([1e-10, 1e-2])
    ax[0].set_box_aspect(1.0)

    ax[1].plot(y, velX, color="black", label="jxf")
    ax[1].plot(y, velX_ref, color="red", linestyle="--", label="ref")
    ax[1].legend()
    ax[1].set_xlabel(r"$y$")
    ax[1].set_ylabel(r"$u$")
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 0.25])
    ax[1].set_box_aspect(1.0)

    fname = rf"Step = {step:03d}, Loss = {loss:3.2e}, $\mu$ = {viscosity:.2f}"
    fig.suptitle(fname)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def main() -> None:
    base_path = Path(__file__).resolve().parent
    output_path = base_path / "pngs"
    output_path.mkdir(exist_ok=True)

    input_manager = InputManager(
        str(base_path / "poiseuille.json"),
        str(base_path / "numerical_setup.json"),
    )
    sim_manager = SimulationManager(input_manager)

    feed_forward = build_feed_forward(sim_manager)
    loss_fn = build_loss_fn(feed_forward)

    domain_information = sim_manager.domain_information
    _, y, _ = domain_information.get_global_cell_centers()
    y = y.squeeze()

    visc_list: list[float] = [0.5, 1.0, 2.0]
    results: dict[float, tuple[Array, Array]] = {}
    for visc in visc_list:
        solution = feed_forward(jnp.array(visc))
        results[visc] = (y, solution)

    create_reference_plot(results, base_path / "poiseuille.png")

    _, velX_ref = results[1.0]

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    params = jnp.array(0.5)

    learning_rate = 1e-1
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    loss_history: list[Array] = []
    total_steps = 125
    steps_vec = np.arange(1, total_steps + 1)
    for i in steps_vec:
        (loss, velX), grad = value_and_grad(params, velX_ref)

        # Update the scalar viscosity from the differentiable profile mismatch.
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        loss_history.append(loss)

        print(f"Step = {i:03d}, Loss = {loss:3.2e}, Visc = {params:.2f}")

        create_optimization_plot(
            y=y,
            velX=velX,
            velX_ref=velX_ref,
            loss_history=loss_history,
            step=i,
            total_steps=total_steps,
            viscosity=params,
            loss=loss,
            save_path=output_path / f"iter_{i:04d}.png",
        )

if __name__ == "__main__":
    main()
