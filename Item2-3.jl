using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots

# Defining and plotting the ODE

# Constants
G = 6.674e-11      # Gravitational constant (m³ kg⁻¹ s⁻²)
c = 3.0e8          # Speed of light (m/s)
a = 7.56e-16       # Radiation constant (J m⁻³ K⁻⁴)
pi = 3.1416

# Polytropic equation parameters
n = 3  # Polytropic index (n = 3 for a Sun-like star)
K = 1.0e13  # Polytropic constant (varies by star type)

# Opacity (assuming as constant)
kappa = 0.2  # Opacity (m²/kg)

# Function to get density from pressure using polytropic equation of state
function density(P)
    return (P / K)^(n / (n + 1))  # rho = (P/K)^(n/(n+1))
end

# Function to estimate temperature from pressure (ideal gas approximation)
function compute_temperature(P, rho)
    μ = 0.6  # Mean molecular weight (for Sun-like star)
    k_B = 1.38e-23  # Boltzmann constant (J/K)
    m_H = 1.67e-27  # Proton mass (kg)
    return (P * μ * m_H) / (rho * k_B)  # T = P / (ρ R), with R = k_B / (μ m_H)
end

# Initial conditions (at center r=0)
P_c = 1.0e16  # Initial pressure
rho_c = density(P_c)  # Computing initial density
T_c = compute_temperature(P_c, rho_c)  # Computing initial temperature
M0 = 0.0      # Initial Mass
L0 = 1.0e26   # Initial luminosity 

u0 = [M0, L0, P_c, T_c]  # Initial conditions [M, L, P, T]

datasize = 30
r_span = (1e6, 1e9) # Starting from a small nonzero radius else for 0, solver step crashes
r_steps = range(r_span[1], r_span[2]; length = datasize)  # Discretization

# Define the true stellar structure ODE function to generate training data
function trueODEfunc(du, u, p, r)
    M, L, P, T = u  
    rho = density(P)  
    
    # Mass Continuity Equation
    du[1] = 4 * pi * r^2 * rho  # dM/dr

    # Energy Generation Equation (assuming epsilon as a constant)
    epsilon = 1.0e5  # J/kg/s
    du[2] = 4 * pi * r^2 * rho * epsilon  # dL/dr

    # Hydrostatic Equilibrium Equation
    du[3] = - (G * M * rho) / r^2  # dP/dr

    #  Energy Transport (Radiative Transfer Equation)
    du[4] = - (3 * kappa * rho * L) / (16 * pi * a * c * r^2 * T^3)  # dT/dr
end

prob_trueode = ODEProblem(trueODEfunc, u0, r_span)

ode_data = Array(solve(prob_trueode, Rosenbrock23(), reltol=1e-4, abstol=1e-4, saveat=r_steps))


p1 = plot(ode_data[1, :], title="Mass vs Steps", label="M", xlabel="Steps", ylabel="Mass")
p2 = plot(ode_data[2, :], title="Luminosity vs Steps", label="L", xlabel="Steps", ylabel="Luminosity")
p3 = plot(ode_data[3, :], title="Pressure vs Steps", label="P", xlabel="Steps", ylabel="Pressure")
p4 = plot(ode_data[4, :], title="Temperature vs Steps", label="T", xlabel="Steps", ylabel="Temperature")

plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))

# Define New Neural ODE model
rng = Random.default_rng()
dudt2 = Lux.Chain(Lux.Dense(4, 50, tanh), Lux.Dense(50, 4)) # 4 i/p, o/p since 4 variables M, L, P, T
# dudt2 = Lux.Chain(Lux.Dense(4, 50, swish), Lux.Dense(50, 50, swish), Lux.Dense(50, 4))

p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, r_span, Rosenbrock23(); saveat = r_steps)

# Define Prediction function
function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

# # Define loss function
# function loss_neuralode(p)
#     pred = predict_neuralode(p)
#     pred_norm = (pred .- data_mean) ./ data_std  # Normalize predictions
#     loss = sum(abs2, ode_data_norm .- pred_norm) / length(ode_data)
#     return loss, pred
# end

function loss_neuralode(p)
    pred = predict_neuralode(p)

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    norm_factors = [maximum(abs, ode_data[i, :]) + epsilon for i in 1:4]

    # Compute relative loss
    loss = sum(sum(abs2, (ode_data[i, :] .- pred[i, :]) ./ norm_factors[i]) for i in 1:4) / length(r_steps)

    return loss, pred
end

using Zygote
using LinearAlgebra 
# Callback for training visualization
callback = function (p, l, pred; doplot=true)
    grad = Zygote.gradient(x -> loss_neuralode(x)[1], p)[1]  # Compute gradient
    grad_norm = norm(grad)  # Compute norm of gradient
    println("Loss: ", l, " | Grad Norm: ", grad_norm)
    if doplot
        plt1 = scatter(r_steps, ode_data[1, :]; label="Data (M)")
        scatter!(plt1, r_steps, pred[1, :]; label="Prediction (M)")
        plt2 = scatter(r_steps, ode_data[2, :]; label="Data (L)")
        scatter!(plt2, r_steps, pred[2, :]; label="Prediction (L)")
        plt3 = scatter(r_steps, ode_data[3, :]; label="Data (P)")
        scatter!(plt3, r_steps, pred[3, :]; label="Prediction (P)")
        plt4 = scatter(r_steps, ode_data[4, :]; label="Data (T)")
        scatter!(plt4, r_steps, pred[4, :]; label="Prediction (T)")
        display(plot(plt1, plt2, plt3, plt4, layout=(2,2)))
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot = true)

# Optimization setup
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 300)


optprob2 = remake(optprob; u0 = result_neuralode.u)
result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01); callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)