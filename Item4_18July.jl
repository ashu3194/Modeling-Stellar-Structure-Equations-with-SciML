using ComponentArrays, Lux, OrdinaryDiffEq, Random
using DiffEqFlux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using CairoMakie

# --- 1. Physical Parameters ---
const G = 6.674e-11
const c = 3.0e8
const a = 7.56e-16
const pi = 3.1416
const n = 3
const K = 1.0e13
const kappa = 0.2

function safe_density(P)
    P = max(P, 1e-12)
    return (P / K)^(n / (n + 1))
end

function compute_temperature(P, rho)
    μ = 0.6
    k_B = 1.38e-23
    m_H = 1.67e-27
    return (P * μ * m_H) / (rho * k_B)
end

function stellar_structure!(du, u, p, r)
    # Clamp state for safety
    M, L, P, T = max.(u, [0.0, 0.0, 1e-12, 1.0])
    rho = safe_density(P)
    du[1] = 4 * pi * r^2 * rho
    epsilon = 1.0e5
    du[2] = 4 * pi * r^2 * rho * epsilon
    du[3] = - (G * M * rho) / max(r^2, 1.0)
    du[4] = - (3 * kappa * rho * L) / (16 * pi * a * c * max(r^2, 1.0) * max(T^3, 1.0))
end

# --- 2. Generate Synthetic Data (True Trajectories) ---
P_c = 1.0e16
rho_c = safe_density(P_c)
T_c = compute_temperature(P_c, rho_c)
M0 = 0.0
L0 = 1.0e26
u0 = [M0, L0, P_c, T_c]
rspan = (1e6, 1e9)
t = range(rspan[1], rspan[2], length=25)

prob = ODEProblem(stellar_structure!, u0, rspan)
sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4, saveat=t)
data = Array(sol)
t = sol.t

# --- 3. Split Data for Training ---
datasize = div(length(t), 2)
tsteps = t[1:datasize]
data_train = data[:, 1:datasize]

# --- 4. Data Normalization (per-variable) ---
using Statistics
μ = mean(data_train, dims=2)
σ = std(data_train, dims=2) .+ 1e-8                    # Small fudge for safety
data_train_norm = (data_train .- μ) ./ σ

# --- 5. Define Neural Networks with tiny outputs ---
rng = MersenneTwister(123)
function make_nn()
    Lux.Chain(
        Lux.Dense(2, 10, tanh),
        Lux.Dense(10, 1),
        x -> 1e-3 * x # very small (almost off) corrections initially
    )
end
NN_M, NN_L, NN_P, NN_T = make_nn(), make_nn(), make_nn(), make_nn()
p_M, st_M = Lux.setup(rng, NN_M)
p_L, st_L = Lux.setup(rng, NN_L)
p_P, st_P = Lux.setup(rng, NN_P)
p_T, st_T = Lux.setup(rng, NN_T)
p0_vec = (layer_M = p_M, layer_L = p_L, layer_P = p_P, layer_T = p_T)
p0_vec = ComponentArray(p0_vec)

# --- 6. UDE-augmented stellar structure (with CLAMPING) ---
function stellar_structure_ude!(du, u, p, r)
    # Clamp state for physical safety
    M, L, P, T = max.(u, [0.0, 0.0, 1e-12, 1.0])
    rho = safe_density(P)
    # Neural corrections, tightly clamped
    dM = clamp(NN_M([r, rho], p.layer_M, st_M)[1][1], -1e3, 1e3)
    dL = clamp(NN_L([r, rho], p.layer_L, st_L)[1][1], -1e3, 1e3)
    dP = clamp(NN_P([r, rho], p.layer_P, st_P)[1][1], -1e3, 1e3)
    dT = clamp(NN_T([r, rho], p.layer_T, st_T)[1][1], -1e3, 1e3)
    du[1] = 4 * pi * r^2 * rho + dM
    epsilon = 1.0e5
    du[2] = 4 * pi * r^2 * rho * epsilon + dL
    du[3] = - (G * M * rho) / max(r^2, 1.0) + dP
    du[4] = - (3 * kappa * rho * L) / (16 * pi * a * c * max(r^2, 1.0) * max(T^3, 1.0)) + dT
end

prob_pred = ODEProblem(stellar_structure_ude!, u0, (tsteps[1], tsteps[end]))

# --- 7. Loss, Prediction (normalized) ---
function predict_stellar(θ)
    Array(solve(prob_pred, Rosenbrock23(), p=θ, saveat=tsteps,
                reltol=1e-4, abstol=1e-4,
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function predict_norm(θ)
    x = predict_stellar(θ)
    x_norm = (x .- μ) ./ σ
    x_norm
end

function loss_stellar(θ)
    x_norm = predict_norm(θ)
    sum(abs2, data_train_norm .- x_norm)
end

# --- 8. Training ---
losses_stellar = Float64[]
iter_stellar = 0
function callback_stellar(θ, l)
    push!(losses_stellar, l)
    global iter_stellar
    iter_stellar += 1
    if iter_stellar % 50 == 0
        println("Iter $iter_stellar | Normalized Loss: $l")
    end
    return false
end

iterations = 500
adtype = Optimization.AutoZygote()
optf_stellar = Optimization.OptimizationFunction((x, p) -> loss_stellar(x), adtype)
optprob_stellar = Optimization.OptimizationProblem(optf_stellar, p0_vec)

# ADAM phase
res_stellar_adam = Optimization.solve(optprob_stellar, OptimizationOptimisers.ADAM(0.001),
                                      callback = callback_stellar, maxiters = iterations)
println("\nFinal normalized training loss after $(length(losses_stellar)) (ADAM): $(losses_stellar[end])")

# BFGS phase
optprob_stellar_bfgs = Optimization.OptimizationProblem(optf_stellar, res_stellar_adam.u)
res_stellar_bfgs = Optimization.solve(optprob_stellar_bfgs, Optim.BFGS(initial_stepnorm = 0.01),
                                      callback = callback_stellar, allow_f_increases = false)
println("\nFinal normalized training loss after $(length(losses_stellar)) (BFGS): $(losses_stellar[end])")

# --- 9. Prediction and Results Visualization ---
prediction = predict_stellar(res_stellar_bfgs.u)

fig2 = CairoMakie.Figure(size=(900,900))
labels = ["Mass", "Luminosity", "Pressure", "Temperature"]
for (i, label) in enumerate(labels)
    ax = CairoMakie.Axis(fig2[div(i-1,2)+1, mod(i-1,2)+1], title=label)
    CairoMakie.scatter!(ax, tsteps, data_train[i,:], color=:black, label="True")
    CairoMakie.lines!(ax, tsteps, prediction[i,:], color=:red, label="Prediction")
    CairoMakie.axislegend(ax)
end
CairoMakie.save("stellar_prediction_vs_true.png", fig2)
println("\nComparison plot saved as stellar_prediction_vs_true.png")

# Final MSE per variable (in original scale)
for i in 1:4
    println("MSE for $(labels[i]): ", mean(abs2, data_train[i,:] .- prediction[i,:]))
end