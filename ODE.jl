using DifferentialEquations
using Plots

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

#= 
    Define the ODE system 
    u = [M; L; P; T]
    u[1] = M(r), pressure (Pa)
    u[2] = L(r), enclosed mass (kg)
    u[3] = P(r), temperature (K)
    u[4] = T(r), luminosity (W)
=#
function stellar_structure!(du, u, p, r)
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

# Initial conditions (at center r=0)
P_c = 1.0e16  # Initial pressure
rho_c = density(P_c)  # Computing initial density
T_c = compute_temperature(P_c, rho_c)  # Computing initial temperature
M0 = 0.0      # Initial Mass
L0 = 1.0e26   # Initial luminosity 

u0 = [M0, L0, P_c, T_c]  # Initial conditions [M, L, P, T]

# Radius span
# rspan(0,R), R=10^9
r_span = (1e6, 1e9)  # Starting from a small nonzero radius else for 0, solver step crashes

# Define the ODE Problem and solve the ODE
prob = ODEProblem(stellar_structure!, u0, r_span)

# Tried solvers like TRBDF2, KenCarp3, Rodas4, Tsit5, but they are mostly giving error for rescode: DtLessThanMin or unstable
# Tried to alter the tolerance values also, but the most data points(vector elements) are coming for Rosenbrock23, for mostly the datapoints 
# and graphs were very sharp. Rosenbrock23 gave a smooth graph
sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)

# Extract solutions
radius = sol.t
mass = sol[1, :]
luminosity = sol[2, :]
pressure = sol[3, :]
temperature = sol[4, :]

# Plot results
p_mass = plot(radius, mass, xlabel="Radius (m)", ylabel="Mass (kg)", title="Mass vs Radius", legend=false, grid=true, yformatter=:scientific)
p_luminosity = plot(radius, luminosity, xlabel="Radius (m)", ylabel="Luminosity (W)", title="Luminosity vs Radius", legend=false, grid=true, yformatter=:scientific)
p_pressure = plot(radius, pressure, xlabel="Radius (m)", ylabel="Pressure (Pa)", title="Pressure vs Radius", legend=false, grid=true, yformatter=:scientific)
p_temperature = plot(radius, temperature, xlabel="Radius (m)", ylabel="Temperature (K)", title="Temperature vs Radius", legend=false, grid=true, yformatter=:scientific)

# Combine plots
plot(p_mass, p_luminosity, p_pressure, p_temperature, layout=(2, 2), size=(1200, 800))