from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from jaxfluids_thirdparty.exact_riemann_solver.helper_functions import \
    get_f_K_df_K, get_Q_K, EOS, speed_of_sound

class ExactRiemannSolver:
    '''Exact Riemann solver using the Stiffend Equation of State
    Based on Ivings et al. - 1998 - On Riemann solvers for compressible liquids
    '''

    def __init__(self) -> None:
        pass

    def solve(
            self,
            W_L: List[float],
            W_R: List[float],
            gamma_L: float = 1.4,
            gamma_R: float = 1.4,
            pb_L: float = 0.0,
            pb_R: float = 0.0,
            Nx: int = 100,
            ts: float = 1.0,
            xlims: Tuple = (-1, 1),
            x0: float = 0.0,
            init_guess: str = "AM",
            max_iters: int = 100,
            tol: float = 1e-06,
            is_verbose: bool = False,
            is_visualize: bool = False
        ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Exact solution of the given Riemann problem
        
        :param W_L: List of primitive left states (density, velocity, pressure)
        :type W_L: List[float]
        :param W_R: List of primitive right states (density, velocity, pressure)
        :type W_R: List[float]
        :param gamma_L: ratio of specific heats left state, defaults to 1.4
        :type gamma_L: float, optional
        :param gamma_R: ratio of specific heats right state, defaults to 1.4
        :type gamma_R: float, optional
        :param pb_L: background pressure left state, defaults to 0.0
        :type pb_L: float, optional
        :param pb_R: background pressure right state, defaults to 0.0
        :type pb_R: float, optional
        :param Nx: number of grind points, defaults to 100
        :type Nx: int, optional
        :param ts: sampling time, defaults to 1.0
        :type ts: float, optional
        :param xlims: domain limits, defaults to (-1, 1)
        :type xlims: Tuple, optional
        :param x0: position of initial discontinuity, defaults to 0.0
        :type x0: float, optional
        :param init_guess: Initial guess for Newton's method, defaults to 'TR'
        :type init_guess: str, optional
        :param max_iters: Number of maximum iterations for Newtons's method, defaults to 100
        :type max_iters: int, optional
        :param tol: Tolerance for Newton's method, defaults to 1e-06
        :type tol: float, optional
        :param is_verbose: Flag for printing information, defaults to True
        :type is_verbose: bool, optional
        :param is_visualize: Flag for plotting results, defaults to True
        :type is_visualize: bool, optional
        :return: pressure in star region, velocity in star region, solution vector, spatial grid vector
        :rtype: Tuple[float, float, np.ndarray, np.ndarray]
        """

        rho_L, u_L, p_L = W_L
        rho_R, u_R, p_R = W_R
        if is_verbose:
            print('\n\n' + '-'*50 + '\n' + '-'*50 + '\nSolving Riemann problem with initial states:')
            print('W_L =', W_L)
            print('W_R =', W_R)

        a_L = speed_of_sound(p_L, rho_L, gamma_L, pb_L)
        a_R = speed_of_sound(p_R, rho_R, gamma_R, pb_R)

        p_star, u_star, iters, err, p_init = self.compute_star_state(
            rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R,
            gamma_L, gamma_R, pb_L, pb_R, init_guess, max_iters, tol)

        if is_verbose:
            print('\nSolving for star states:')
            print('p0 = %4.4f, iters = %i, err = %4.3e' %(p_init, iters, err))
            print('u* = %4.4f, p* = %4.4f' %(u_star, p_star))

        # Sample the solution and do plotting
        x = np.linspace(xlims[0], xlims[1], Nx)
        x_mapped = np.linspace(xlims[0] - x0, xlims[1] - x0, Nx)
        W_init = self.sample_solution(
            x_mapped, 0.0, p_star, u_star,
            rho_L, u_L, p_L,
            rho_R, u_R, p_R,
            a_L, a_R, gamma_L, gamma_R, pb_L, pb_R)
        W = self.sample_solution(
            x_mapped, ts, p_star, u_star,
            rho_L, u_L, p_L,
            rho_R, u_R, p_R,
            a_L, a_R, gamma_L, gamma_R,
            pb_L, pb_R)
        if is_visualize:
            fig2 = self.plot_solution(x, W, W_init)
            plt.show()

        return p_star, u_star, W, x

    def compute_star_state(
            self, 
            rho_L: float,
            u_L: float,
            p_L: float,
            a_L: float, 
            rho_R: float,
            u_R: float,
            p_R: float,
            a_R: float,
            gamma_L: float,
            gamma_R: float,
            pb_L: float,
            pb_R: float,
            init_guess: str,
            max_iters: int,
            tol: float
            ) -> Tuple[float, float, int, float, float]:
        '''
        Solve for p_star and u_star with Newton method as outlined in Toro Chap. 4.2 / 4.3
        1) Solve f(p, W_L, W_R) = f_L(p, W_L) + f_R(p, W_R) + (u_R - u_L) == 0 for p_star
        2) u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R(p_star) + f_L(p_star))
        '''
        alpha_L = 2 / ((gamma_L + 1) * rho_L)
        alpha_R = 2 / ((gamma_R + 1) * rho_R)
        beta_L = p_L * (gamma_L - 1) / (gamma_L + 1)
        beta_R = p_R * (gamma_R - 1) / (gamma_R + 1)
        
        # Initial guess for pressure
        if init_guess == 'TR':
            ''' Toro - Eq. 4.46 '''
            raise NotImplementedError
            # p_init = ( ( a_L + a_R - 0.5 * (gamma - 1) * (u_R - u_L) ) / ( a_L / p_L**g5 + a_R / p_R**g5 ) )**(1/g5)
        elif init_guess == 'PV':
            ''' Toro - Eq. 4.47 '''
            p_init = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
        elif init_guess == 'TS':
            ''' Toro - Eq. 4.48 '''
            p_hat = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
            g_L = np.sqrt(alpha_L / (p_hat + beta_L))
            g_R = np.sqrt(alpha_R / (p_hat + beta_R))
            p_init = ( g_L * p_L + g_R * p_R - (u_R - u_L) ) / (g_L + g_R)
        elif init_guess == 'AM':
            ''' Toro - Eq. 4.49 '''
            p_init = 0.5 * (p_L + p_R) 

        p_old = max(tol, p_init)

        # Newtons method
        CHA = 1e+08
        iters = 0
        while CHA > tol and iters < max_iters:
            iters += 1

            f_L, df_L = get_f_K_df_K(p_old, a_L, rho_L, p_L, gamma_L, pb_L)
            f_R, df_R = get_f_K_df_K(p_old, a_R, rho_R, p_R, gamma_R, pb_R)
            f = f_L + f_R + (u_R - u_L)
            df = df_L + df_R

            ''' Toro - Eq. 4.44'''
            p_new = p_old - f / df
            CHA = np.abs(0.5*(p_new-p_old)/(p_new+p_old))
            p_old = max(tol, p_new)   # to avoid negative pressures
        p_star = p_old
        u_star = 0.5 * (u_L + u_R + f_R - f_L)
        return p_star, u_star, iters, CHA, p_init

    def sample_solution(
            self,
            x: np.ndarray,
            t: float,
            p_star: float,
            u_star: float,
            rho_L: float,
            u_L: float,
            p_L: float, 
            rho_R: float,
            u_R: float,
            p_R: float, 
            a_L: float,
            a_R: float,
            gamma_L: float,
            gamma_R: float,
            pb_L: float,
            pb_R: float
            ) -> np.ndarray:
        '''
        Sample the solution over domain x at time t
        '''
        
        a_star_L = a_L * ((p_star + pb_L)/(p_L + pb_L))**((gamma_L - 1)/(2 * gamma_L))
        a_star_R = a_R * ((p_star + pb_R)/(p_R + pb_R))**((gamma_R - 1)/(2 * gamma_R))

        Q_L = get_Q_K(p_star, rho_L, p_L, gamma_L, pb_L)
        Q_R = get_Q_K(p_star, rho_R, p_R, gamma_R, pb_R)

        # speed
        S_vec = x / (t + 1e-10)

        S_L = u_L - Q_L / rho_L
        S_HL = u_L - a_L
        S_TL = u_star - a_star_L

        S_R = u_R + Q_R / rho_R
        S_HR = u_R + a_R
        S_TR = u_star + a_star_R

        W_mat = np.zeros((4, S_vec.shape[0]))

        for i, S in enumerate(S_vec):
            # Solution left of contact discontinuity
            if S < u_star:
                # Left fan
                if p_star < p_L:
                    if S < S_HL:
                        # Left of the head of the rarefaction wave
                        W = [rho_L, u_L, p_L]
                    elif S > S_TL:
                        # Right of the tail of the rarefaction wave
                        rho_star_L = rho_L * ((p_star + pb_L) / (p_L + pb_L))**(1/gamma_L)  # Isentropic relation
                        W = [rho_star_L, u_star, p_star]
                    else:
                        # Inside rarefactioon wave
                        rho_L_fan = rho_L * (2/(gamma_L+1) + (gamma_L-1)/(gamma_L+1)/a_L*(u_L - S))**(2/(gamma_L-1))
                        u_L_fan = 2/(gamma_L+1) * (a_L + (gamma_L-1)/2*u_L + S)
                        p_L_fan = (p_L + pb_L) * (2/(gamma_L+1) + (gamma_L-1)/(gamma_L+1)/a_L*(u_L - S))**(2*gamma_L/(gamma_L-1)) - pb_L
                        W = [rho_L_fan, u_L_fan, p_L_fan]
                # Left shock
                else:
                    if S < S_L:
                        # Left of the shock
                        W = [rho_L, u_L, p_L]
                    else:
                        # Right of the shock
                        rho_star_L = rho_L*((p_star/p_L+(gamma_L-1)/(gamma_L+1))/((gamma_L-1)/(gamma_L+1)*p_star/p_L +1))
                        W = [rho_star_L, u_star, p_star]
            # Solution right of contact discontinuity
            else:
                # Right shock
                if p_star > p_R:
                    if S < S_R:
                        # Left of the shock
                        rho_star_R = rho_R*((p_star/p_R+(gamma_R-1)/(gamma_R+1))/((gamma_R-1)/(gamma_R+1)*p_star/p_R +1))
                        W = [rho_star_R, u_star, p_star]
                    else:
                        # Right of the shock
                        W = [rho_R, u_R, p_R]
                # Right fan
                else:
                    if S < S_TR:
                        # Left of the tail of the rarefaction wave
                        rho_star_R = rho_R * ((p_star + pb_R) / (p_R + pb_R))**(1/gamma_R)  # Isentropic relation
                        W = [rho_star_R, u_star, p_star]
                    elif S > S_HR:
                        # Right of the head of the rarefaction wave
                        W = [rho_R, u_R, p_R]
                    else:
                        # Inside the rarefaction wave
                        rho_R_fan = rho_R * (2/(gamma_R+1) - (gamma_R-1)/(gamma_R+1)/a_R*(u_R - S))**(2/(gamma_R-1))
                        u_R_fan = 2/(gamma_R+1) * (-a_R + (gamma_R-1)/2*u_R + S)
                        p_R_fan = (p_R + pb_R) * (2/(gamma_R+1) - (gamma_R-1)/(gamma_R+1)/a_R*(u_R - S))**(2*gamma_R/(gamma_R-1)) - pb_R
                        W = [rho_R_fan, u_R_fan, p_R_fan]
            W_mat[:3, i] = np.array(W)
            # W_mat[3, i] = get_energy_from_specific(W[0], W[1], EOS(W[2], W[0], gamma))
            if S < u_star:
                W_mat[3, i] = EOS(W[2], W[0], gamma_L, pb_L)
            else:
                W_mat[3, i] = EOS(W[2], W[0], gamma_R, pb_R)

        return W_mat

    def plot_solution(
            self,
            x: np.ndarray,
            W: np.ndarray,
            W_init: np.ndarray = None
            ):
        fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True)
        ax = ax.flatten()
        for i, ax_i in enumerate(ax):
            ax_i.plot(x, W[i, :])
            if W_init.any():
                ax_i.plot(x, W_init[i, :], linestyle=':', color='gray')
        for ax_i, label_i in zip(ax, [r'$\rho$', r'$u$', r'$p$', r'$e$']):
            ax_i.set_ylabel(label_i)
            ax_i.set_xlim(x[0], x[-1])
        ax[2].set_xlabel(r'$x$')
        ax[3].set_xlabel(r'$x$')
        plt.tight_layout()

        return fig
