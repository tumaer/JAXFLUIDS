from functools import partial
from typing import Callable

import jax
# jax.config.update("jax_enable_x64", True)
from jax._src.lax import linalg as lax_linalg
import jax.numpy as jnp
from jax import Array
import numpy as np

class BDF_Solver:

    def __init__(
            self,
            ode_fun: Callable,
            batch_size: int,
            num_odes: int,
            jac_fun: Callable = None,
            atol: float = 1e-6,
            rtol: float = 1e-3,
        ) -> None:

        self.max_order = 5
        self.orders = np.arange(0, self.max_order+1)

        # RECIPROCAL SUMS
        self.gamma_k = np.concatenate([[np.nan], np.cumsum(1. / self.orders[1:])])
        # BDF coefficients - appended a 0.0 at 0-th index to make indexing with "order" easier
        self.kappa_coeff = np.array([0.0, -0.1850, -1/9, -0.0823, -0.0415, 0.0]) 
        self.newton_coefficients = 1. / ((1 - self.kappa_coeff) * self.gamma_k)
        self.error_coefficients = self.kappa_coeff * self.gamma_k + 1.0 / (self.orders + 1)

        # print("RECIPROCAL SUMS:", self.gamma_k)
        # print("NEWTON COEFFS:", self.newton_coefficients)
        # print("ERROR COEFFS:", self.error_coefficients)

        self.rtol               = rtol
        self.atol               = atol
        self.newton_tol_factor  = 0.1
        self.safety_factor      = 0.9
        self.epsilon            = 1e-12

        self.MAX_NEWTON_ITERS = 4
        self.newton_step_size_factor = 0.5
        self.min_step_size_factor    = 0.1
        self.max_step_size_factor    = 10

        self.batch_size = batch_size
        self.num_odes   = num_odes
        self.ode_fun    = ode_fun
        self.jac_fun    = jac_fun

    # NOTE: has to jitted for performance! 
    @partial(jax.jit, static_argnums=(0))
    def solve(
            self,
            initial_time: float,
            initial_state: Array,
            t_final: float
        ):
        # print("NUM ODES:", self.num_odes)
        # print("BATCH SIZE:", self.batch_size)
        # print("INTIAL CONDITION:", initial_state)

        current_time  = initial_time * jnp.ones(self.batch_size, dtype=jnp.float64)
        initial_step_size = self.initial_step_size(initial_state, initial_time, self.ode_fun)
        is_integration_finished   = (current_time == t_final)
        num_steps                 = jnp.zeros(self.batch_size, dtype=jnp.int32)
        num_steps_same_size       = jnp.zeros(self.batch_size, dtype=jnp.int32)
        should_update_jacobian    = jnp.ones(self.batch_size, dtype=bool)
        should_update_step_size   = jnp.zeros(self.batch_size, dtype=bool)
        order                     = jnp.ones(self.batch_size, dtype=jnp.int32)
        new_step_size = step_size = initial_step_size

        # ARRAY OF BACKWARD DIFFERENCES
        first_order_backward_difference = initial_step_size[:, None] * self.ode_fun(initial_time, initial_state)
        backward_differences = jnp.concatenate([
            initial_state[:, None], 
            first_order_backward_difference[:, None], 
            jnp.zeros((self.batch_size, self.max_order + 1, self.num_odes), dtype=jnp.float64)
            ], axis=1)

        # print("CURRENT TIME:", current_time)
        # print("INITIAL STEP SIZE:", initial_step_size)
        # print("INITIAL BACKDIFF ARRAY:", backward_differences.shape)

        # NOTE cond_fun must return a boolean scalar, but got output type(s) [ShapedArray(bool[2])].
        def step_cond(states):
            return jnp.logical_not(states["is_integration_finished"].all())

        states = {
            "is_integration_finished"   : is_integration_finished,
            "t_final"                   : t_final,
            "current_time"              : current_time,
            "backward_differences"      : backward_differences,
            "step_size"                 : step_size,
            "new_step_size"             : new_step_size,
            "should_update_step_size"   : should_update_step_size,
            "num_steps"                 : num_steps,
            "num_steps_same_size"       : num_steps_same_size,
            "order"                     : order,
            "jacobian_mat"              : jnp.zeros((self.batch_size, self.num_odes, self.num_odes), dtype=jnp.float64),
            "is_accepted"               : jnp.zeros((self.batch_size), dtype=bool)
        }

        sols = [states["backward_differences"][:,0]]
        times = [states["current_time"]]

        states = jax.lax.while_loop(step_cond, self.step, states)

        # ii = 0

        # while step_cond(states):
        #     states = self.step(states)
        #     print("SOLVE: CURRENT TIME :", states["current_time"])
        #     # print("SOLVE: TIME STEP    :", states["new_step_size"])
        #     print("SOLVE: ORDER        :", states["order"])
        #     print("SOLVE: NUM INT STEPS:", states["num_steps"])
        #     print("SOLVE: INT FINISHED :", states["is_integration_finished"])
        #     input()
        #     print("SOLVE: CURRENT STATE:", states["backward_differences"][:,0])
        #     input()
        #     if ii % 50 == 0:
        #         sols.append(states["backward_differences"][:,0])
        #         times.append(states["current_time"])

        #     states["current_time"] = jnp.where((states_["current_time"] < t_final), states_["current_time"], states["current_time"])
        #     states["current_state"] = jnp.where((states_["current_time"] < t_final)[:,None], states_["current_state"], states["current_state"])
        #     states["backward_differences"] = jnp.where((states_["current_time"] < t_final)[:,None,None], states_["backward_differences"], states["backward_differences"])
        #     sols.append(states["current_state"])
        #     times.append(states["current_time"])
        #     input()

        sols.append(states["backward_differences"][:,0])
        times.append(states["current_time"])

        return jnp.array(sols), jnp.array(times), (states["num_steps"], states["order"])

    @partial(jax.jit, static_argnums=(0))
    def step(self, states):
        t_final       = states["t_final"]
        current_state = states["backward_differences"][:,0]
        current_time  = states["current_time"]
        new_step_size = states["new_step_size"] 
        should_update_step_size = states["should_update_step_size"]
        print("NEW STEP SIZE:", new_step_size)

        distance_to_final_time = t_final - current_time
        is_overstepped = new_step_size > distance_to_final_time
        new_step_size  = jnp.where(is_overstepped, distance_to_final_time, new_step_size)
        print("DISTANCE TO FINAL:", distance_to_final_time)
        print("IS OVERSTEPPED:", is_overstepped)
        print("NEW STEP SIZE:", new_step_size)
        should_update_step_size = should_update_step_size | is_overstepped

        jacobian_mat = self.jacobian_fn(self.ode_fun, current_time, current_state) if self.jac_fun is None else self.jac_fun(current_time, current_state)

        states["new_step_size"] = new_step_size
        states["should_update_step_size"] = should_update_step_size
        states["jacobian_mat"] = jacobian_mat
        states["is_accepted"]  = jnp.where(states["is_integration_finished"], jnp.ones((self.batch_size), dtype=bool), jnp.zeros((self.batch_size), dtype=bool))

        print(jacobian_mat.shape, new_step_size.shape, should_update_step_size.shape)

        def maybe_step_cond(states):
            return jnp.logical_not(states["is_accepted"].all())

        states = jax.lax.while_loop(maybe_step_cond, self.maybe_step, states)

        # while maybe_step_cond(states):
        #     print("    STEP: IS ACCEPTED:", states["is_accepted"])
        #     print("    STEP: STEP SIZE  :", states["new_step_size"])
        #     states = self.maybe_step(states)

        states["is_integration_finished"] = (states["current_time"] == states["t_final"])
        states["num_steps"] += jnp.where(states["is_integration_finished"], 0, 1)

        return states

    @partial(jax.jit, static_argnums=(0))
    def maybe_step(self, states):
        print("MAYBE STEP")
        is_jacobian_up_to_date = True 

        step_size               = states["step_size"] 
        new_step_size           = states["new_step_size"] 
        current_time            = states["current_time"] 
        should_update_step_size = states["should_update_step_size"]
        backward_differences    = states["backward_differences"]
        order                   = states["order"]
        num_steps_same_size     = states["num_steps_same_size"]
        jacobian_mat            = states["jacobian_mat"]

        backward_differences = jnp.where(
            should_update_step_size[:, None, None], 
            self._interpolate_backward_differences(
                    backward_differences, order, new_step_size / step_size),
            backward_differences
        )

        step_size = jnp.where(should_update_step_size, new_step_size, step_size)
        num_steps_same_size = jnp.where(should_update_step_size, 0, num_steps_same_size)

        newton_coefficients = jnp.sum(jnp.where(jnp.arange(self.max_order+1) == order[:,None], self.newton_coefficients, 0.0), axis=1)

        lu, perm = self.newton_lu(jacobian_mat, newton_coefficients, step_size)

        tol = self.atol + self.rtol * jnp.abs(backward_differences[:,0])
        newton_tol = self.newton_tol_factor * jnp.linalg.norm(tol, axis=-1)

        is_newton_converged, next_backward_difference, next_state, _ = self._newton(
            self.ode_fun, 
            backward_differences, 
            order, newton_coefficients,
            lu,
            perm, 
            current_time, 
            step_size, 
            newton_tol)

        # print("CURRENT TIME:", current_time, "STEP SIZE:", step_size)
        # print("BACK DIFF:", backward_differences)
        # print("NEXT STATE:", next_state, "NEXT BACK DIFF:", next_backward_difference)
        # input()

        # If Newton's method failed but Jacobian was up to date --> smaller step size
        is_newton_failed = jnp.logical_not(is_newton_converged)
        should_update_step_size = is_newton_failed & is_jacobian_up_to_date
        new_step_size = step_size * jnp.where(should_update_step_size, self.newton_step_size_factor, 1.0)

        # If Newton's method failed but Jacobian was NOT up to date --> update Jacobian
        # should_update_jacobian = is_newton_failed & (not is_jacobian_up_to_date)
        error_coefficients = jnp.sum(jnp.where(jnp.arange(self.max_order+1) == order[:,None], self.error_coefficients, 0.0), axis=1)

        error_ratio = jnp.where(
            is_newton_converged,
            self.error_ratio(
                next_backward_difference, error_coefficients, tol
            ),
            jnp.nan,
        )
        is_accepted = error_ratio < 1.0
        is_converged_and_rejected = is_newton_converged & jnp.logical_not(is_accepted)

        print("        MAYBE STEP: ORDER      :", order)
        print("        MAYBE STEP: NEXT BACK  :", next_backward_difference)
        print("        MAYBE STEP: NEXT STATE :", next_state)
        print("        MAYBE STEP: ERROR RATIO:", error_ratio)

        # print("NEWTON CONVERGED:", is_newton_converged)
        # print("IS ACCEPTED:", is_accepted)
        # input()

        # If Newton's method converged but the solution was NOT accepted, decrease
        # the step size.
        new_step_size = jnp.where(
            is_converged_and_rejected,
            self.next_step_size(step_size, order, error_ratio),
            new_step_size,
        )
        should_update_step_size = should_update_step_size | is_converged_and_rejected

        # If Newton's method converged and the solution was accepted, update the
        # matrix of backward differences.
        current_time = jnp.where(is_accepted, current_time + step_size, current_time)

        backward_differences = jnp.where(
            is_accepted[:,None,None],
            self.update_backward_differences(backward_differences, next_backward_difference, next_state, order),
            backward_differences,
        )

        is_jacobian_up_to_date = is_jacobian_up_to_date & jnp.logical_not(is_accepted)
        
        num_steps_same_size = jnp.where(
            is_accepted, num_steps_same_size + 1, num_steps_same_size
        )

        # Order and step size are only updated if we have taken strictly more than
        # order + 1 steps of the same size. This is to prevent the order from
        # being throttled.
        should_update_order_and_step_size = is_accepted & (num_steps_same_size > order + 1)

        new_order = order
        new_error_ratio = error_ratio
        for offset in [-1, +1]:
            proposed_order = jnp.clip(order + offset, 1, self.max_order)
            error_coefficients = jnp.sum(jnp.where(jnp.arange(self.max_order+1) == proposed_order[:,None], self.error_coefficients, 0.0), axis=1)
            
            backwards_difference_at_proposed_order = jnp.sum(
                jnp.where(proposed_order[:,None,None] + 1 == jnp.arange(self.max_order + 3).reshape(1,-1,1), backward_differences, 0.0), axis=1
            )
            
            proposed_error_ratio = self.error_ratio(backwards_difference_at_proposed_order, error_coefficients, tol)
            proposed_error_ratio_is_lower = proposed_error_ratio < new_error_ratio
            new_order = jnp.where(
                should_update_order_and_step_size & proposed_error_ratio_is_lower,
                proposed_order,
                new_order,
            )
            new_error_ratio = jnp.where(
                should_update_order_and_step_size & proposed_error_ratio_is_lower,
                proposed_error_ratio,
                new_error_ratio,
            )
        # TODO
        order = new_order
        error_ratio = new_error_ratio

        new_step_size = jnp.where(
            should_update_order_and_step_size,
            self.next_step_size(step_size, order, error_ratio),
            new_step_size,
        )
        should_update_step_size = (
            should_update_step_size | should_update_order_and_step_size
        )

        states["step_size"]                 = jnp.where(jnp.logical_not(states["is_accepted"]), step_size                           , states["step_size"])
        states["new_step_size"]             = jnp.where(jnp.logical_not(states["is_accepted"]), new_step_size                       , states["new_step_size"])
        states["current_time"]              = jnp.where(jnp.logical_not(states["is_accepted"]), current_time                        , states["current_time"])
        states["should_update_step_size"]   = jnp.where(jnp.logical_not(states["is_accepted"]), should_update_step_size             , states["should_update_step_size"])
        states["backward_differences"]      = jnp.where(jnp.logical_not(states["is_accepted"])[:,None,None], backward_differences   , states["backward_differences"])
        states["order"]                     = jnp.where(jnp.logical_not(states["is_accepted"]), order                               , states["order"])
        states["num_steps_same_size"]       = jnp.where(jnp.logical_not(states["is_accepted"]), num_steps_same_size                 , states["num_steps_same_size"])
        states["is_accepted"]               = jnp.where(jnp.logical_not(states["is_accepted"]), is_accepted                         , states["is_accepted"])

        return states

    @partial(jax.jit, static_argnums=(0,1))
    def _newton(
            self,
            ode_fun,
            backward_differences,
            order,
            newton_coefficient,
            lu,
            perm,
            current_time,
            step_size,
            tol
        ):
        """Runs Newton's method to solve the BDF equation."""
        print("NEWTON")
        initial_guess = jnp.sum(
            jnp.where(
                jnp.arange(self.max_order + 1).reshape(1, -1, 1) <= order[:, None, None],
                backward_differences[:, : self.max_order + 1],
                jnp.zeros_like(backward_differences)[:, : self.max_order + 1],
            ),
            axis=1,
        )

        rhs_constant_term = newton_coefficient[:,None] * jnp.sum(
            jnp.where(
                jnp.arange(1, self.max_order + 1).reshape(1, -1, 1) <= order[:, None, None],
                self.gamma_k[None, 1:, None] * backward_differences[:, 1 : self.max_order + 1],
                jnp.zeros_like(backward_differences)[:, 1 : self.max_order + 1],
            ),
            axis=1,
        )

        # print("ORDER:", order)
        # print("BACKWARD DIFF:", backward_differences)
        # print("INITIAL GUESS:", initial_guess)
        # print("STEP SIZE:", step_size)
        # input()

        next_time = current_time + step_size

        def newton_body(iterand):
            """Performs one iteration of Newton's method."""
            print("NEWTON BODY")
            next_backward_difference = iterand["next_backward_difference"]
            next_state               = iterand["next_state"]

            rhs = (
                newton_coefficient[:,None] * step_size[:,None] * ode_fun(next_time, next_state)
                - rhs_constant_term
                - next_backward_difference
            )

            # print(upper.shape, unitary.shape, rhs.shape)

            # delta = jax.scipy.linalg.solve_triangular(
            #         upper, jnp.einsum("ijk, ij -> ik", unitary, rhs), lower=False
            #     )
            delta = lax_linalg.lu_solve(
                    lu, perm, rhs
                )

            num_iters = iterand["num_iters"] + 1

            next_backward_difference += delta
            next_state               += delta

            # TODO: ord=1 might work better ( & faster), originally ord=2
            delta_norm = jnp.linalg.norm(delta, axis=1)
            lipschitz_const = delta_norm / iterand["prev_delta_norm"]

            # Stop if method has converged.
            # approx_dist_to_sol = lipschitz_const / (1.0 - lipschitz_const) * delta_norm
            approx_dist_to_sol = delta_norm / (1.0/lipschitz_const - 1.0) # TODO: this form works better!
            close_to_sol = approx_dist_to_sol < tol
            delta_norm_is_zero = jnp.equal(delta_norm, jnp.array(0.0, dtype=jnp.float64))
            # print(delta_norm_is_zero.shape)
            converged = close_to_sol | delta_norm_is_zero
            finished = converged

            # Stop if any of the following conditions are met:
            # (A) We have hit the maximum number of iterations.
            # (B) The method is converging too slowly.
            # (C) The method is not expected to converge.
            too_slow = lipschitz_const > 1.0
            # print(finished, too_slow, lipschitz_const)
            finished = finished | too_slow

            too_many_iters = jnp.equal(num_iters, self.MAX_NEWTON_ITERS)
            num_iters_left = self.MAX_NEWTON_ITERS - num_iters
            wont_converge = approx_dist_to_sol * lipschitz_const ** num_iters_left > tol

            # print(finished, too_many_iters, wont_converge, self.MAX_NEWTON_ITERS)
            finished = finished | too_many_iters | wont_converge

            # print("    DELTA:", delta)
            # print("    APPROX DISTANCE:", approx_dist_to_sol)
            # print("    NEWTON TOL:", tol)
            # print("    CONVERGED:", converged)
            # print("    FINISHED:", finished)
            # input()

            # converged                = jnp.where(iterand["finished"], iterand["converged"], converged)
            # finished                 = jnp.where(iterand["finished"], iterand["finished"], finished)
            # next_backward_difference = jnp.where(iterand["finished"], iterand["next_backward_difference"], next_backward_difference)
            # next_state               = jnp.where(iterand["finished"], iterand["next_state"], next_state)
            # delta_norm               = jnp.where(iterand["finished"], iterand["prev_delta_norm"], delta_norm)

            return {
                "converged"                 : converged,
                "finished"                  : finished, 
                "next_backward_difference"  : next_backward_difference,   
                "next_state"                : next_state,
                "num_iters"                 : num_iters,
                "prev_delta_norm"           : delta_norm,
            }

        iterand = {
            "converged"                 : jnp.zeros(backward_differences.shape[0], dtype=bool),
            "finished"                  : jnp.zeros(backward_differences.shape[0], dtype=bool),
            "next_backward_difference"  : jnp.zeros_like(initial_guess),
            "next_state"                : initial_guess,
            "num_iters"                 : jnp.zeros(backward_differences.shape[0], dtype=jnp.int32),
            "prev_delta_norm"           : -jnp.zeros(backward_differences.shape[0], dtype=jnp.float64), # TODO: pyBaMM uses -1.0
        }

        # NOTE: prev_delta_norm = -0.0, so that lipschitz_const is -inf and too_slow is False
        # on first iteration

        iterand = jax.lax.while_loop(
            lambda iterand: jnp.logical_not(iterand["finished"].all()), newton_body, iterand
        )

        # while jnp.logical_not(iterand["finished"].all()):
        #     iterand = newton_body(iterand)

        ## Krishna: need to double check this
        return iterand["converged"], iterand["next_backward_difference"], iterand["next_state"], iterand["num_iters"],

    @partial(jax.jit, static_argnums=(0))
    def newton_lu(
            self,
            jacobian_mat,
            newton_coefficient,
            step_size
        ):
        """Computes the LU decomposition of (I - h/(1-kappa)/gamma_k * J)
        Note: jacobian_mat should be of shape (..., N, N)
        jacobian_mat.shape       = (N_samples, N_odes, N_odes)
        newton_coefficient.shape = (N_samples,)
        """
        identity = jnp.eye(self.num_odes)
        newton_mat = identity - step_size[:,None,None] * newton_coefficient[:,None,None] * jacobian_mat

        lu, _, perm = lax_linalg.lu(newton_mat)
        return lu, perm

#  @partial(jax.jit, static_argnums=(0))
#     def newton_qr(self, jacobian_mat, newton_coefficient, step_size):
#         """Computes the QR decomposition of (I - h/(1-kappa)/gamma_k * J)
#         Note: jacobian_mat should be of shape (..., N, N)
#         jacobian_mat.shape       = (N_samples, N_odes, N_odes)
#         newton_coefficient.shape = (N_samples,)
#         """
#         identity = jnp.eye(self.num_odes)
#         newton_mat = identity - step_size[:,None,None] * newton_coefficient[:,None,None] * jacobian_mat
#         q, r = jax.scipy.linalg.qr(newton_mat)
#         return q, r

    @partial(jax.jit, static_argnums=(0, 1))
    def jacobian_fn(
            self,
            ode_fun,
            current_time,
            current_state
            ):
        J = []
        for i in range(self.num_odes):
            current_state_ = current_state
            current_state_ = current_state_.at[:,i].add(self.epsilon)
            dfdy_i = (ode_fun(current_time, current_state_) - ode_fun(current_time, current_state)) / self.epsilon
            J.append(dfdy_i)
        J = jnp.stack(J, axis=2)
        # J = jnp.array([[-1., -2.], [-3., -4.]])
        return J

    @partial(jax.jit, static_argnums=(0))
    def _interpolation_matrix(
            self,
            order,
            step_size_ratio
        ) -> Array:
        """Creates the matrix used to interpolate backward differences."""
        orders = jnp.arange(1, self.max_order + 1)
        i = orders[None, :, None]
        j = orders[None, None, :]
        # Matrix whose (i, j)-th entry (`1 <= i, j <= order`) is
        # `1/j! (0 - i * step_size_ratio) * ... * ((j-1) - i * step_size_ratio)`.

        full_interpolation_matrix = jnp.cumprod(((j - 1) - i * step_size_ratio[:, None, None]) / j, axis=2)

        zeros_matrix = jnp.zeros_like(full_interpolation_matrix)
        interpolation_matrix_ = jnp.where(
            jnp.arange(1, self.max_order + 1) <= order[:, None, None],
            jnp.transpose(
                jnp.where(
                    jnp.arange(1, self.max_order + 1) <= order[:, None, None],
                    jnp.transpose(full_interpolation_matrix, axes=(0,2,1)),
                    zeros_matrix,
                ), axes=(0,2,1)
            ),
            zeros_matrix,
        )
        return interpolation_matrix_

    @partial(jax.jit, static_argnums=(0))
    def _interpolate_backward_differences(
            self,
            backward_differences, 
            order, 
            step_size_ratio
        ) -> Array:
        """Updates backward differences when a change in the step size occurs."""
        interpolation_matrix_ = self._interpolation_matrix(order, step_size_ratio)
        interpolation_matrix_unit_step_size_ratio = self._interpolation_matrix(order, jnp.ones_like(step_size_ratio))

        interpolated_backward_differences_orders_one_to_five = jnp.einsum(
            "ijk, ikl -> ijl",
            interpolation_matrix_unit_step_size_ratio,
            jnp.einsum("ijk, ikl -> ijl", interpolation_matrix_, backward_differences[:, 1 : self.max_order+1]),
        )

        interpolated_backward_differences = jnp.concatenate(
            [
                backward_differences[:,0:1],
                interpolated_backward_differences_orders_one_to_five,
                jnp.zeros((self.batch_size, 2, self.num_odes), dtype=jnp.float64)
            ],
            axis=1,
        )

        return interpolated_backward_differences

    @partial(jax.jit, static_argnums=(0))
    def update_backward_differences(
            self,
            backward_differences,
            next_backward_difference,
            next_state_vec,
            order
        ):
        orders_ = jnp.arange(self.max_order + 3).reshape(1,-1,1)
        new_backward_differences = jnp.zeros_like(backward_differences)
        backward_differences_ = next_backward_difference - jnp.sum(jnp.where(order[:,None,None] + 1 == orders_, backward_differences, 0.0), axis=1) 
        new_backward_differences = jnp.where(order[:,None,None] + 2 == orders_, backward_differences_[:,None,:], new_backward_differences)
        new_backward_differences = jnp.where(order[:,None,None] + 1 == orders_, next_backward_difference[:,None,:], new_backward_differences)
        
        def body(vals):
            k, new_backward_differences_array_ = vals

            new_backward_differences_array_k = jnp.sum(
                jnp.where(orders_ == k[:,None,None], backward_differences, 0.0) \
                + jnp.where(orders_ == k[:,None,None]+1, new_backward_differences_array_, 0.0), 
                axis=1, keepdims=True)

            new_backward_differences_array_ = jnp.where(orders_ == k[:,None,None], new_backward_differences_array_k, new_backward_differences_array_)

            return (k - 1, new_backward_differences_array_)

        def body_cond(vals):
            k, _ = vals
            return (k > 0).any()

        _, new_backward_differences = jax.lax.while_loop(
            body_cond, body, (order, new_backward_differences)
        )

        new_backward_differences = new_backward_differences.at[:,0].set(next_state_vec)
        new_backward_differences = jnp.where(order[:,None,None] + 3 <= orders_, 0.0, new_backward_differences)

        return new_backward_differences

    @partial(jax.jit, static_argnums=(0,3))
    def initial_step_size(
            self,
            initial_state,
            initial_time,
            ode_fun,
            max_step_size=1.,
            min_step_size=1e-12,
        ):
        first_derivative = ode_fun(initial_time, initial_state)
        # Naive prediction: y_{eps} = y_0 + eps * F(t_0, y_0)
        next_state = initial_state + self.epsilon * first_derivative
        next_time  = initial_time + self.epsilon

        second_derivative = (ode_fun(next_time, next_state) - first_derivative) / self.epsilon 
        tol = self.atol + self.rtol * jnp.abs(initial_state)

        norm = jnp.linalg.norm(self.error_coefficients[1] * second_derivative / tol, axis=-1)
        initial_step_size = jax.lax.rsqrt(norm)
        return jnp.clip(self.safety_factor * initial_step_size, min_step_size, max_step_size)

    @partial(jax.jit, static_argnums=(0))
    def next_step_size(
            self,
            step_size,
            order,
            error_ratio
        ) -> float:
        factor = error_ratio**(-1. / (order + 1.))
        next_step_size = step_size * jnp.clip(self.safety_factor * factor, self.min_step_size_factor, self.max_step_size_factor)
        return next_step_size

    @partial(jax.jit, static_argnums=(0))
    def error_ratio(
            self,
            backward_difference,
            error_coefficient,
            tol
        ):
        """Computes the ratio of the error in the computed state to the tolerance."""
        error_ratio_ = jnp.linalg.norm(error_coefficient[:,None] * backward_difference / tol, axis=-1)
        return error_ratio_
