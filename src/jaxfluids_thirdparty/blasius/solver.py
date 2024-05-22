from typing import Tuple

import numpy as np

# function definition for Blasius similarity solution
def solve_compressible_blasius(
		Minf: float = 2.0,
		Gamma: float = 1.4,
		Pr: float = 1.0,
		Tinf: float = 300.0,
		is_adiabatic: bool = False,
		Twall: float = 2.0,
		C2: float = 0.0,
		lim: float = 8.0,
		N: int = 500,
		delta: float = 1e-10,
		eps: float = 1e-9,
		) -> Tuple:
	"""Iterative solver for the compressible laminar Blasius equations.

	Oz, Furkan, and Kursat Kara. "A CFD tutorial in Julia: Introduction 
	to compressible laminar boundary-layer flows." Fluids 6.11 (2021): 400.
	
	:param Minf: Mach number, defaults to 2
	:type Minf: float, optional
	:param Gamma: Ratio of specific heats, defaults to 1.4
	:type Gamma: float, optional
	:param Pr: Prandtl number, defaults to 1.0
	:type Pr: float, optional
	:param Tinf: Free stream temeperature, defaults to 300.0
	:type Tinf: float, optional
	:param is_adiabatic: Boolean indicating adiabatic wall, defaults to False
	:type is_adiabatic: bool, optional
	:param Twall: Wall temperature, defaults to 2.0
	:type Twall: float, optional
	:param C2: Sutherland coefficient in Kelvin, defaults to 0.0, defaults to 0.0
	:type C2: float, optional
	:param lim: The value which simulates lim-> inf, defaults to 10.0
	:type lim: float, optional
	:param N: Number of	points, defaults to 500
	:type N: int, optional
	:param delta: small number for shooting method, defaults to 1e-10
	:type delta: float, optional
	:param eps: Tolerance for convergence, defaults to 1e-9
	:type eps: float, optional
	:return: eta, y, velocityX, temperature
	:rtype: Tuple
	"""
	
	h = lim / N # Delta	y

	# Initializing
	y1 = np.zeros([N + 1, 1]) # f
	y2 = np.zeros([N + 1, 1]) # f'
	y3 = np.zeros([N + 1, 1]) # f''
	y4 = np.zeros([N + 1, 1]) # rho(eta)
	y5 = np.zeros([N + 1, 1]) # rho(eta)'
	eta = np.arange(0, lim + h, h) # Iteration of eta up to	infinity dalfa = 0

	dalpha = 0
	dbeta = 0

	if is_adiabatic:
		# Boundary Conditions for Adiabatic Case
		y1[0] = 0
		y2[0] = 0
		y5[0] = 0

		# Initial Guess	for the beginning of simulation
		alpha0 = 0.1 # Initial Guess
		beta0 = 3 # Initial Guess
	else:
		# Boundary Conditions for Isothermal Case
		y1[0] = 0
		y2[0] = 0
		y4[0] = Twall

		# Initial Guess for Beginning of Simulation
		alpha0 = 0.1 # Initial Guess
		beta0 = 3 # Initial Guess

	for ite in range(100000):
		if is_adiabatic:
			# Boundary Conditions for Adiabatic Case
			y1[0] = 0
			y2[0] = 0
			y5[0] = 0

			y3[0] = alpha0
			y4[0] = beta0
		else:
			# Boundary Conditions for Isothermal Case
			y1[0] = 0
			y2[0] = 0
			y4[0] = Twall

			y3[0] = alpha0
			y5[0] = beta0

		[y1, y2, y3, y4, y5] = RK(eta, h, y1, y2, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma)

		y2old = y2[-1].copy()
		y4old = y4[-1].copy()

		if is_adiabatic:
			# Boundary Conditions for Adiabatic Case
			y1[0] = 0
			y2[0] = 0
			y5[0] = 0

			y3[0] = alpha0 + delta
			y4[0] = beta0
		else:
			# Boundary Conditions for Isothermal Case
			y1[0] = 0
			y2[0] = 0
			y4[0] = Twall

			y3[0] = alpha0 + delta
			y5[0] = beta0

		[y1, y2, y3, y4, y5] = RK(eta, h, y1, y2, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma)

		y2new1 = y2[-1].copy()
		y4new1 = y4[-1].copy()

		if is_adiabatic:
			# Boundary Conditions for Adiabatic Case
			y1[0] = 0
			y2[0] = 0
			y5[0] = 0

			y3[0] = alpha0
			y4[0] = beta0 + delta
		else:
			# Boundary Conditions for Isothermal Case
			y1[0] = 0
			y2[0] = 0
			y4[0] = Twall

			y3[0] = alpha0
			y5[0] = beta0 + delta

		[y1, y2, y3, y4, y5] = RK(eta, h, y1, y2, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma)

		y2new2 = y2[-1].copy()
		y4new2 = y4[-1].copy()

		a11 = (y2new1 - y2old) / delta
		a21 = (y4new1 - y4old) / delta
		a12 = (y2new2 - y2old) / delta
		a22 = (y4new2 - y4old) / delta
		r1 = 1 - y2old
		r2 = 1 - y4old
		dalpha = (a22 * r1 - a12 * r2) / (a11 * a22 - a12 * a21)
		dbeta = (a11 * r2 - a21 * r1) / (a11 * a22 - a12 * a21)
		alpha0 = alpha0 + dalpha
		beta0 = beta0 + dbeta
	
		if (abs(y2[-1] - 1) < eps) and (abs(y4[-1] - 1) < eps):
			Truey2 = y2[-1].copy()
			Truey4 = y4[-1].copy()
			break

	# xaxis = np.zeros([len(eta), 1])

	# for i in range(1, len(eta)):
	# 	xaxis[i] = (eta[i] - 0) * (y4[0] + 2 * np.sum(y4[1:i-1])+y4[i]) / (2 * eta[i]) * h

	y = np.zeros(len(eta))
	for i in range(1,len(eta)):
		y[i] = y[i-1] + y4[i]*(eta[i]-eta[i-1])

	xaxis = y*np.sqrt(2)

	return eta, xaxis, y2, y4


def Y1(y2):
	return y2

def Y2(y3):
	return y3

def Y3(y1, y3, y4, y5, C2, Tinf):
	RHS = -y3*((y5/(2*(y4)))-(y5/(y4+C2/Tinf)))-y1*y3*((y4+C2/Tinf)/(np.sqrt(y4)*(1+C2/Tinf)))
	return RHS

def Y4(y5):
	return y5

def Y5(y1, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma):
	RHS = -y5**2 * ((0.5 / y4) - (1 / (y4 + C2 / Tinf)))-Pr * y1 * y5 / np.sqrt(y4) * (y4 + C2 / Tinf) / (1 + C2 / Tinf)-(Gamma - 1) * Pr * Minf**2 * y3**2
	return RHS


def RK(eta, h, y1, y2, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma):
	for i in range(len(eta) - 1):
		k11 = Y1(y2[i])
		k21 = Y2(y3[i])
		k31 = Y3(y1[i], y3[i], y4[i], y5[i], C2, Tinf)
		k41 = Y4(y5[i])
		k51 = Y5(y1[i], y3[i], y4[i], y5[i], C2, Tinf, Minf, Pr, Gamma)

		k12 = Y1(y2[i] + 0.5 * h * k21)
		k22 = Y2(y3[i] + 0.5 * h * k31)
		k32 = Y3(y1[i] + 0.5 * h * k11, y3[i] + 0.5 * h * k31, y4[i] + 0.5 * h * k41, y5[i] + 0.5 * h * k51, C2, Tinf)
		k42 = Y4(y5[i] + 0.5 * h * k51)
		k52 = Y5(y1[i] + 0.5 * h * k11, y3[i] + 0.5 * h * k31, y4[i] + 0.5 * h * k41, y5[i] + 0.5 * h * k51, C2, Tinf, Minf, Pr, Gamma)

		k13 = Y1(y2[i] + 0.5 * h * k22)
		k23 = Y2(y3[i] + 0.5 * h * k32)
		k33 = Y3(y1[i] + 0.5 * h * k12, y3[i] + 0.5 * h * k32, y4[i] + 0.5 * h * k42, y5[i] + 0.5 * h * k52, C2, Tinf)
		k43 = Y4(y5[i] + 0.5 * h * k52)
		k53 = Y5(y1[i] + 0.5 * h * k12, y3[i] + 0.5 * h * k32, y4[i] + 0.5 * h * k42, y5[i] + 0.5 * h * k52, C2, Tinf, Minf, Pr, Gamma)

		k14 = Y1(y2[i] + h * k23)
		k24 = Y2(y3[i] + h * k33)
		k34 = Y3(y1[i] + h * k13, y3[i] + h * k33, y4[i] + h * k43, y5[i] + h * k53, C2, Tinf)
		k44 = Y4(y5[i] + h * k53)
		k54 = Y5(y1[i] + h * k13, y3[i] + h * k33, y4[i] + h * k43, y5[i] + h * k53, C2, Tinf, Minf, Pr, Gamma)

		y5[i + 1] = y5[i] + (1 / 6) * (k51 + 2 * k52 + 2 * k53 + k54) * h
		y4[i + 1] = y4[i] + (1 / 6) * (k41 + 2 * k42 + 2 * k43 + k44) * h
		y3[i + 1] = y3[i] + (1 / 6) * (k31 + 2 * k32 + 2 * k33 + k34) * h
		y2[i + 1] = y2[i] + (1 / 6) * (k21 + 2 * k22 + 2 * k23 + k24) * h
		y1[i + 1] = y1[i] + (1 / 6) * (k11 + 2 * k12 + 2 * k13 + k14) * h
	return [y1, y2, y3, y4, y5]
