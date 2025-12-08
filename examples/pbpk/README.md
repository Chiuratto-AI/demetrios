# PBPK ODE Solver in Demetrios

ODE solver with Euler and RK4 methods for 1-compartment PK model.

## Model

dA/dt = -k_el * A

## Parameters

- Dose: 500 mg
- Volume: 50 L
- k_el: 0.1 /h
- Step size: 0.5 h

## Usage

dc compile ode_solver.d
