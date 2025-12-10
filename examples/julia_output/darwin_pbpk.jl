# Darwin PBPK - Generated from Demetrios
# Single and multi-compartment PBPK model for Julia

using DifferentialEquations
using Unitful
using Plots

# Parameter structure
@kwdef struct PBPKParams
    v_blood::Float64 = 5.0
    v_liver::Float64 = 1.8
    v_kidney::Float64 = 0.31
    cl_hepatic::Float64 = 30.0
    cl_renal::Float64 = 2.0
    fu::Float64 = 0.03
    kp_liver::Float64 = 3.5
    kp_kidney::Float64 = 2.8
end

# Single-compartment IV bolus
function pbpk_iv!(du, u, p::PBPKParams, t)
    c = u[1]
    k = p.cl_hepatic / p.v_blood * p.fu
    du[1] = -k * c
end

# 3-compartment model
function pbpk_3comp!(du, u, p::PBPKParams, t)
    c_blood = u[1]
    c_liver = u[2]
    c_kidney = u[3]
    
    k = p.cl_hepatic / p.v_blood * p.fu
    du[1] = -k * c_blood
    du[2] = p.kp_liver * du[1]
    du[3] = p.kp_kidney * du[1]
end

# Solve IV bolus
function simulate_iv(dose::Float64; tspan=(0.0, 2.0))
    p = PBPKParams()
    u0 = [dose / p.v_blood]
    prob = ODEProblem(pbpk_iv!, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-9)
    return sol
end

# Solve 3-compartment
function simulate_3comp(dose::Float64; tspan=(0.0, 2.0))
    p = PBPKParams()
    u0 = [dose / p.v_blood, 0.0, 0.0]
    prob = ODEProblem(pbpk_3comp!, u0, tspan, p)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-9)
    return sol
end

# Main
dose = 2.0
tspan = (0.0, 2.0)

sol = simulate_iv(dose, tspan=tspan)
c_final = sol[1, end]
println("IV Bolus - Final concentration:  mg/L")
println("Expected (theory): 0.01191 mg/L")
