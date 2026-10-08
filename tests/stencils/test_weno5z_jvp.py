"""Tests for the custom JVP of the WENO5-Z cell-face value.

In near-flat stencils the stock tangent sum_k (d omega_k p_k + omega_k d p_k)
cancels terms of size ~1/delta and jax.jvp is not linear in the tangent.
These tests check that the custom JVP rule
  - leaves the primal bitwise unchanged (vs a frozen copy of the stock code),
  - is linear in the tangent (additivity and homogeneity),
  - agrees with the stock JVP on smooth data,
  - is consistent with central finite differences (O(h^2)),
  - transposes correctly (dot-product test <w, J v> = <J^T w, v>),
  - produces finite JVPs and gradients.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jaxfluids.stencils.reconstruction.shock_capturing.weno5_base import WENO5Base  # noqa: E402
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno5_z import WENO5Z  # noqa: E402

N = 128
NH = 3
SEEDS = range(10)
LIN_TOL = 1e-13


class _StockWENO5Z(WENO5Base):
    """Frozen copy of the WENO5-Z reconstruction before the custom JVP (no custom derivative)."""

    def __init__(self, nh, inactive_axes, offset=0, **kwargs):
        super().__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        self.debug = False

    def reconstruct_xi(self, buffer, axis, j, dx=None, **kwargs):
        s1_ = self.s_[j][axis]
        u_imm = buffer[s1_[0]]
        u_im = buffer[s1_[1]]
        u_i = buffer[s1_[2]]
        u_ip = buffer[s1_[3]]
        u_ipp = buffer[s1_[4]]

        beta_0, beta_1, beta_2 = self.smoothness(u_imm, u_im, u_i, u_ip, u_ipp)
        tau_5 = jnp.abs(beta_0 - beta_2)

        alpha_z_0 = self._dr[0] * (1.0 + tau_5 / (beta_0 + self.eps))
        alpha_z_1 = self._dr[1] * (1.0 + tau_5 / (beta_1 + self.eps))
        alpha_z_2 = self._dr[2] * (1.0 + tau_5 / (beta_2 + self.eps))

        one_alpha_z = 1.0 / (alpha_z_0 + alpha_z_1 + alpha_z_2)

        omega_z_0 = alpha_z_0 * one_alpha_z
        omega_z_1 = alpha_z_1 * one_alpha_z
        omega_z_2 = alpha_z_2 * one_alpha_z

        p_0, p_1, p_2 = self.polynomials(u_imm, u_im, u_i, u_ip, u_ipp)
        return omega_z_0 * p_0 + omega_z_1 * p_1 + omega_z_2 * p_2


def _faces(stencil):
    """u (1D buffer incl. halos) -> both cell-face values (j = 0, 1) along x, flattened."""
    def f(u):
        b = u[:, None, None]
        return jnp.concatenate([stencil.reconstruct_xi(b, 0, 0).ravel(),
                                stencil.reconstruct_xi(b, 0, 1).ravel()])
    return f


def _line(kind, seed=0):
    L = N + 2 * NH
    x = (np.arange(L) - NH + 0.5) / N
    rng = np.random.default_rng(1000 + seed)
    if kind == "flat":
        # near-flat: 0.4 plus relative noise at 1e-14, with a smooth bump
        u = 0.4 + 0.2 * np.exp(-((x - 0.7) / 0.06) ** 2) + 1e-14 * 0.4 * rng.standard_normal(L)
    elif kind == "smooth":
        u = 0.4 + 0.2 * np.sin(2 * np.pi * x)
    elif kind == "step":
        u = np.where(x < 0.5, 1.0, 0.125) + 1e-14 * rng.standard_normal(L)
    else:
        raise ValueError(kind)
    return jnp.asarray(u, dtype=jnp.float64)


def _unit(rng, shape):
    v = rng.standard_normal(shape)
    return jnp.asarray(v / np.linalg.norm(v), dtype=jnp.float64)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


@pytest.fixture(scope="module")
def faces():
    patched = _faces(WENO5Z(nh=NH, inactive_axes=["y", "z"]))
    stock = _faces(_StockWENO5Z(nh=NH, inactive_axes=["y", "z"]))
    return patched, stock


def _jvp(f, u):
    g = jax.jit(lambda uu, v: jax.jvp(f, (uu,), (v,))[1])
    return lambda v: g(u, v)


@pytest.mark.parametrize("kind", ["flat", "smooth", "step"])
def test_primal_bitwise(faces, kind):
    fp, fs = faces
    u = _line(kind)
    np.testing.assert_array_equal(np.asarray(fp(u)), np.asarray(fs(u)))
    np.testing.assert_array_equal(np.asarray(jax.jit(fp)(u)), np.asarray(jax.jit(fs)(u)))


@pytest.mark.parametrize("kind", ["flat", "smooth", "step"])
def test_jvp_linear(faces, kind):
    fp, _ = faces
    for seed in SEEDS:
        u = _line(kind, seed)
        jv = _jvp(fp, u)
        rng = np.random.default_rng(seed)
        v, w = _unit(rng, u.shape), _unit(rng, u.shape)
        Jv, Jw = jv(v), jv(w)
        assert _rel(jv(v + w), Jv + Jw) <= LIN_TOL
        for a in (3.0, 1e-3, 1e2):
            assert _rel(jv(a * v), a * Jv) <= LIN_TOL


def test_jvp_matches_stock_on_smooth(faces):
    fp, fs = faces
    u = _line("smooth")
    for seed in SEEDS:
        v = _unit(np.random.default_rng(100 + seed), u.shape)
        assert _rel(_jvp(fp, u)(v), _jvp(fs, u)(v)) <= 1e-12


def test_jvp_central_fd_second_order(faces):
    fp, _ = faces
    # smooth line and smooth direction: a random direction makes the FD stencil cross the
    # kinks of |beta_0 - beta_2| and the WENO-Z switching, which is not a JVP defect
    u = _line("smooth")
    x = (np.arange(u.shape[0]) - NH + 0.5) / N
    c = np.cos(6 * np.pi * x)
    v = jnp.asarray(c / np.linalg.norm(c))
    Jv = _jvp(fp, u)(v)
    scale = float(jnp.max(jnp.abs(u))) / float(jnp.max(jnp.abs(v)))
    errs = []
    for e in (1e-2, 1e-3):
        h = e * scale
        fd = (fp(u + h * v) - fp(u - h * v)) / (2 * h)
        errs.append(_rel(fd, Jv))
    assert errs[1] < 1e-8
    assert errs[0] / errs[1] > 50.0  # O(h^2): ratio ~100 for a factor 10 in h


@pytest.mark.parametrize("kind", ["flat", "smooth", "step"])
def test_dot_product(faces, kind):
    fp, _ = faces
    u = _line(kind)
    y = fp(u)
    jv = _jvp(fp, u)
    vjp = jax.jit(lambda w: jax.vjp(fp, u)[1](w)[0])
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        v, w = _unit(rng, u.shape), _unit(rng, y.shape)
        Jv = jv(v)
        lhs = float(jnp.vdot(w, Jv))
        rhs = float(jnp.vdot(vjp(w), v))
        assert abs(lhs - rhs) <= 1e-14 * float(jnp.linalg.norm(w) * jnp.linalg.norm(Jv))


@pytest.mark.parametrize("kind", ["flat", "smooth", "step"])
def test_jvp_and_grad_finite(faces, kind):
    fp, _ = faces
    u = _line(kind)
    v = _unit(np.random.default_rng(3), u.shape)
    assert bool(jnp.all(jnp.isfinite(_jvp(fp, u)(v))))
    g = jax.grad(lambda uu: jnp.sum(fp(uu) ** 2))(u)
    assert bool(jnp.all(jnp.isfinite(g)))
