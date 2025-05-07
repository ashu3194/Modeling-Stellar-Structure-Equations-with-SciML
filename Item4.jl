using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random

# Physical constants
G = 6.674e-11       # Gravitational constant (m³ kg⁻¹ s⁻²)
c = 3.0e8           # Speed of light (m/s)
a = 7.56e-16        # Radiation constant (J m⁻³ K⁻⁴)
pi = 3.1416         # Pi

# Polytropic parameters
n = 3
K = 1.0e13
kappa = 0.2

# Function to get density from pressure
function density(P)
    return (P / K)^(n / (n + 1))
end

# Function to compute temperature from pressure and density
function compute_temperature(P, rho)
    μ = 0.6
    k_B = 1.38e-23
    m_H = 1.67e-27
    return (P * μ * m_H) / (rho * k_B)
end

## Generate synthetic data with stellar structure model (M -> u[1], L -> u[2], P -> u[3], T -> u[4])
function stellar_structure!(du, u, p, r)
    M, L, P, T = u
    rho = density(P)

    du[1] = 4 * pi * r^2 * rho                              # dM/dr
    epsilon = 1.0e5 
    du[2] = 4 * pi * r^2 * rho * epsilon                      # dL/dr
    du[3] = - (G * M * rho) / r^2                           # dP/dr
    du[4] = - (3 * kappa * rho * L) / (16 * pi * a * c * r^2 * T^3)  # dT/dr
end

P_c = 1.0e16
rho_c = density(P_c)
T_c = compute_temperature(P_c, rho_c)
M0 = 0.0
L0 = 1.0e26
u0 = [M0, L0, P_c, T_c]

p0 = [6.674e-11, 3.0e8, 7.56e-16, 0.2, 3, 1.0e13, 1.0e5]
rspan = (1e6, 1e9)

# t = radius not time
t = range(rspan[1], rspan[2], length=25)

prob = ODEProblem(stellar_structure!, u0, rspan, p0)
sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4, saveat=t)

data = Array(sol)
t = sol.t 
using CairoMakie

# Create a figure with a 2x2 grid for four subplots
f = Figure(size = (800, 800))

# Plot Mass vs Radius
ax1 = CairoMakie.Axis(f[1, 1], title = "Mass", xlabel = "Radius", ylabel = "Mass (kg)")
CairoMakie.scatter!(ax1, t, data[1, :], color = :blue)

# Plot Luminosity vs Radius
ax2 = CairoMakie.Axis(f[1, 2], title = "Luminosity", xlabel = "Radius", ylabel = "Luminosity (W)")
CairoMakie.scatter!(ax2, t, data[2, :], color = :orange)

# Plot Pressure vs Radius
ax3 = CairoMakie.Axis(f[2, 1], title = "Pressure", xlabel = "Radius", ylabel = "Pressure (Pa)")
CairoMakie.scatter!(ax3, t, data[3, :], color = :green)

# Plot Temperature vs Radius
ax4 = CairoMakie.Axis(f[2, 2], title = "Temperature", xlabel = "Radius", ylabel = "Temperature (K)")
CairoMakie.scatter!(ax4, t, data[4, :], color = :red)

# Save the figure to a file
save("stellar_separate_plots.png", f)

# Define the training set for the stellar model
tspan_train = (1e6, 1e9)  # Example range for radius, adjust as needed
datasize = length(t[1:Int(round(length(t)*0.50))])  
tsteps = range(tspan_train[1], tspan_train[2]; length = datasize)  # Adjusted for radius
data_train = data[:, 1:datasize]  # Extract the training data for the first 50% of the data

# Neural Networks (to replace interaction terms in the stellar structure model)
rng = Random.default_rng()
Random.seed!(92)  # for reproducibility

# Neural network for mass (M)
NN_M = Lux.Chain(Lux.Dense(2, 10, tanh), Lux.Dense(10, 1))
p_M, st_M = Lux.setup(rng, NN_M)

# Neural network for luminosity (L)
NN_L = Lux.Chain(Lux.Dense(2, 10, tanh), Lux.Dense(10, 1))
p_L, st_L = Lux.setup(rng, NN_L)

# Neural network for pressure (P)
NN_P = Lux.Chain(Lux.Dense(2, 10, tanh), Lux.Dense(10, 1))
p_P, st_P = Lux.setup(rng, NN_P)

# Neural network for temperature (T)
NN_T = Lux.Chain(Lux.Dense(2, 10, tanh), Lux.Dense(10, 1))
p_T, st_T = Lux.setup(rng, NN_T)

# Combine the parameters into a ComponentArray
p0_vec = (layer_M = p_M, layer_L = p_L, layer_P = p_P, layer_T = p_T)
p0_vec = ComponentArray(p0_vec)

# Define the UDE system (with NNs replacing the equations for M, L, P, and T)
function stellar_structure_ude!(du, u, p, r)
    M, L, P, T = u
    rho = density(P)
    
    # Neural network outputs for mass, luminosity, pressure, and temperature
    dM = NN_M([r, rho], p.layer_M, st_M)[1][1]
    dL = NN_L([r, rho], p.layer_L, st_L)[1][1]
    dP = NN_P([r, rho], p.layer_P, st_P)[1][1]
    dT = NN_T([r, rho], p.layer_T, st_T)[1][1]

    # Mass Continuity Equation (replaced by NN output)
    du[1] = 4 * pi * r^2 * rho + dM
    epsilon = 1.0e5
    du[2] = 4 * pi * r^2 * rho * epsilon + dL
    du[3] = - (G * M * rho) / r^2 + dP
    du[4] = - (3 * kappa * rho * L) / (16 * pi * a * c * r^2 * T^3) + dT
end

# Define the ODEProblem
prob_pred = ODEProblem(stellar_structure_ude!, u0, tspan_train)

# Define the prediction function using adjoint sensitivity
function predict_stellar(θ)
    Array(solve(prob_pred, Rosenbrock23(), p=θ, saveat=tsteps,
                reltol=1e-4, abstol=1e-4,
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# Define the loss function
function loss_stellar(θ)
    x = predict_stellar(θ)
    loss = sum(abs2, (data_train .- x))
    return loss
end

losses_stellar = []
iter_stellar = 0

function callback_stellar(θ, l)
    push!(losses_stellar, l)
    global iter_stellar
    iter_stellar += 1
    if iter_stellar % 100 == 0
        println("Iter $iter_stellar | Loss: $l")
    end
    return false
end

iterations = 500
adtype = Optimization.AutoZygote()

# Define optimization function
optf_stellar = Optimization.OptimizationFunction((x, p) -> loss_stellar(x), adtype)
optprob_stellar = Optimization.OptimizationProblem(optf_stellar, p0_vec)

# First phase: ADAM optimizer
res_stellar_adam = Optimization.solve(optprob_stellar, OptimizationOptimisers.ADAM(0.01),
                                      callback = callback_stellar, maxiters = iterations)

println("Final training loss after $(length(losses_stellar)) iterations (ADAM): $(losses_stellar[end])")

# Second phase: BFGS optimizer
optprob_stellar_bfgs = Optimization.OptimizationProblem(optf_stellar, res_stellar_adam.u)
res_stellar_bfgs = Optimization.solve(optprob_stellar_bfgs, Optim.BFGS(initial_stepnorm = 0.01),
                                      callback = callback_stellar, allow_f_increases = false)

println("Final training loss after $(length(losses_stellar)) iterations (BFGS): $(losses_stellar[end])")

# Final prediction
prediction_stellar = predict_stellar(res_stellar_bfgs.u)
