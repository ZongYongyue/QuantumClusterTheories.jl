using QuantumLattices
using QuantumClusterTheories
using ExactDiagonalization
using LinearAlgebra
using Serialization
using Optim
using Distributed
"""
    ExactNormalGreenFunction(t, μ, k, ω; η=0.05)
    ExactSWaveGreenFunction(t, Δ, μ, ϕ, k, ω; η=0.05)
    ExactPxWaveGreenFunction(t, Δ, μ, k, ω; η=0.05)

Analytic expression of some 1d Green function.
"""
function ExactNormalGreenFunction(t, μ, k, ω; η=0.05)
    ep = 2t*cos(k) + μ
    g = 1/(ω - ep + η*im)
    return [g 0 ; 0 g]
end
function  ExactSWaveGreenFunction(t, Δ, μ, ϕ, k, ω; η=0.05)
    ep, pm, pp = 2t*cos(k) + μ, exp(-im*ϕ), exp(im*ϕ)
    xi = √(ep^2 + Δ^2)
    uv, u², v², g, l = abs(Δ)/(2*xi), (1+ep/xi)/2, (1-ep/xi)/2, 1/(ω - xi + η*im), 1/(ω + xi + η*im)
    a = (g - l)
    g11, g14, g32, g33 = u²*g + v²*l, -uv*pm*a, uv*pp*a, v²*g + u²*l
    g22, g23, g41, g44 = g11, -g14, -g32, g33
    return [g11 0 0 g14; 0 g22 g23 0; 0 g32 g33 0; g41 0 0 g44]
end
function  ExactPxWaveGreenFunction(t, Δ, μ, k, ω; η=0.05)
    ep, pm, pp = 2t*cos(k) + μ, -2im*sin(k), 2im*sin(k)
    xi = √(ep^2 + (pm*pp)*Δ^2)
    uv, u², v², g, l = abs(Δ)/(2*xi), (1+ep/xi)/2, (1-ep/xi)/2, 1/(ω - xi + η*im), 1/(ω + xi + η*im)
    a = (g - l)
    g11, g14, g32, g33  = u²*g + v²*l, uv*pm*a, uv*pp*a, v²*g + u²*l
    g22, g23, g41, g44 = g11, g14, g32, g33
    return [g11 0 0 g14; 0 g22 g23 0; 0 g32 g33 0; g41 0 0 g44]
end

"""
    belongs(ids::AbstractVector)

Select certain sites in the cluster.
"""
function belongs(ids::AbstractVector)
    function sublattices(bond::Bond)
        if bond.points[1].site in ids
            return 1.0
        else
            return 0.0
        end
    end
    return sublattices
end

"""
    phase_by_moire(ϕ::Real, spin::Rational)

Phase exp(im*ϕ*σ).
"""
function phase_by_moire(ϕ::Real, spin::Rational)
    function spinup(bond::Bond)
        θ = azimuth(rcoordinate(bond))
        @assert any(≈(θ),(0, 2π/3, 4π/3, π/3, π, 5π/3, 2π)) "Triangle error:wrong input bond."
        any(≈(θ),(0, 2π/3, 4π/3, 2π)) && return exp(-im*ϕ)
        any(≈(θ),(π/3, π, 5π/3)) && return exp(im*ϕ)
    end
    function spindown(bond::Bond)
        θ = azimuth(rcoordinate(bond))
        @assert any(≈(θ),(0, 2π/3, 4π/3, π/3, π, 5π/3, 2π)) "Triangle error:wrong input bond."
        any(≈(θ),(0, 2π/3, 4π/3, 2π)) && return exp(im*ϕ)
        any(≈(θ),(π/3, π, 5π/3)) && return exp(-im*ϕ)
    end
    if spin == 1//2
        return spinup
    elseif spin == -1//2
        return spindown
    else
        return "error"
    end
end

"""
    antiferro(wavevector::AbstractVector)

    The phase that determined by antiferromagnetic order in square lattice.
"""
function antiferro(wavevector::AbstractVector)
    function af(bond::Bond)
        return exp(im*dot(wavevector,rcoordinate(bond)))
    end
    return af
end

"""
    phase_by_azimuth(azs::AbstractVector, phs::AbstractVector)

The phase that determined by azimuth.
"""
function phase_by_azimuth(azs::AbstractVector, phs::AbstractVector)
    function pbz(bond::Bond)
        θ = azimuth(rcoordinate(bond))
        for i in eachindex(azs)
            any(≈(θ), azs[i]) && return phs[i]
        end
    end
    return pbz
end

"""
    coupling_by_spinrotation(θ::Real, ϕ::Real)

The coupling that determined by spinrotation.
"""
function coupling_by_spinrotation(θ::Real, ϕ::Real)
    sin(θ)*cos(ϕ)*MatrixCoupling(:, FID, :, σ"x", :) + sin(θ)*sin(ϕ)*MatrixCoupling(:, FID, :, σ"y", :) + cos(θ)*MatrixCoupling(:, FID, :, σ"z", :)
end

"""
    spawn(numworkers::Int)

Perform multiprocess.
"""
function spawn(numworkers::Int)
    np = length(workers())
    np == 1 ? addworkers=numworkers : (np < numworkers ? addworkers=(numworkers - np) : addworkers = 0)
    addprocs(addworkers)
end


"""
    saveData(data, filename::String) -> .jls

save data(e.g. a VCA data) as a jls file
"""
function saveData(data, filename::String)
    open(filename, "w") do io
        serialize(io, data)
    end
end

"""
    loadData(filename::String)

load data from a jls file
"""
function loadData(filename::String)
    data = nothing
    open(filename, "r") do io
        data = deserialize(io)
    end
    return data
end

function minimax(func::Function, start::AbstractVector, maxstart::Integer, accur::Real=1e-4, max_iteration::Integer=30)
    step = [10*copy(accur) for _ in 1:length(start)]
    steps_min, steps_max = step[1:maxstart-1], step[maxstart:end]
    iter, X0 = 0, start
    options = Optim.Options(x_tol=accur, f_tol=5e-7, iterations=100, show_trace=true)
    while iter < max_iteration 
        x_min, x_max = X0[1:maxstart-1], X0[maxstart:end]
        res₁ = optimize(x->func([x..., x_max...]), x_min, NelderMead(initial_simplex = MiniMaxSimplexer(steps_min)), options)
        res₂ = optimize(x->-func([x_min..., x...]), x_max, NelderMead(initial_simplex = MiniMaxSimplexer(steps_max)), options)
        sol = [res₁.minimizer..., res₂.minimizer...]
        diff = sol - X0
        steps_min, steps_max = abs.(diff[1:maxstart-1]), abs.(diff[maxstart:end])
        X0 = sol
        iter += 1
        any(diff.>accur) ? (converged=false) : (converged=true)
        if converged 
            return (sol, func(sol))
        end
    end
    error("Too many iterations: $max_iteration")
end

struct MiniMaxSimplexer{T<:AbstractVector} <: Optim.Simplexer 
    step::T
end
function Optim.simplexer(S::MiniMaxSimplexer, initial_x::AbstractArray{T, N}) where {T, N}
    n = length(initial_x)
    initial_simplex = [zeros(Float64, n) for _ in 1:n+1]
    for i in 1:n+1
        initial_simplex[i][:] .= initial_x
    end
    for i in 1:n
        initial_simplex[i+1][i] += S.step[i]
    end
    return initial_simplex
end

function Hessian(func, x, step)
    n = length(x)
    N = 1 + 2*n + div((n*(n-1)), 2)
    y, hess, xp = zeros(Float64, N), zeros(Float64, n, n), zeros(Float64, N, n)
    for k in 1:N
        xp[k, :] = copy(x)
    end
    k = 1
    for i in 1:n
        k += 1
        xp[k, i] += step[i]
        k += 1
        xp[k, i] -= step[i]
    end
    for i in 1:n
        for j in 1:i-1
            k += 1
            xp[k, i] += 0.707*(2*((i-1)%2)-1)*step[i]
            xp[k, j] += 0.707*(2*((j-1)%2)-1)*step[j]
        end
    end
    y, A = funcvals(func, xp), zeros(Float64, N, N)
    A[:, 2:n+1] = xp
    for k in 1:N
        A[k, 1] = 1.0
        m = n + 2
        for i in 1:n
            for j in 1:i
                A[k, m] = xp[k, i] * xp[k, j]
                if i == j
                    A[k, m] *= 0.5
                end
                m += 1
            end
        end
    end
    A1 = inv(A)
    y = A1*y
    val = y[1]
    grad = y[2:n+1]
    val += sum(dot(grad,x))
    m = n+2
    for i in 1:n
        for j in 1:i
            hess[i,j] = y[m]
            hess[j,i] = y[m]
            if i == j
                val += x[i]*x[j]*y[m]*0.5
            else
                val += x[i]*x[j]*y[m]
            end
            m += 1
        end
    end
    grad += hess*x
    return val, grad, hess
end

function funcvals(func, x)
    n = size(x, 1)
    y = zeros(Float64, n)
    for i in 1:n
        y[i] = func(x[i,:])
    end
    return y  
end

function NewtonStep(func, x, step)
    F, gradient, hessian = Hessian(func, x, step)
    ihessian = inv(hessian)
    dx = -(ihessian*gradient)
    return dx, F, gradient, ihessian
end

function Newton(func, start, step=[1e-3 for _ in length(start)], accur=[1e-6 for _ in length(start)], gtol=1e-4, max_iteration=30)
    n = length(start)
    gradient = zeros(Float64, n)
    dx = zeros(Float64, n)
    y = zeros(Float64, n)
    iteration = 0
    x = start
    step0 = step
    while iteration < max_iteration
        iteration += 1
        gradient0 = copy(gradient)
        dx, F, gradient, hessian = NewtonStep(func, x, step)
        global ihessian = inv(hessian)
        if norm(gradient) < gtol 
            print("convergence on gradient after ,$iteration, iterations")
            break      
        end             
        x += dx
        step_multiplier = 2.0
        for i in 1:n
            step[i] = abs(dx[i])
            if step[i] > step0[i]
                step[i] = 2.0*step0[i]
            end
            if step[i] < step_multiplier*accur[i]
                step[i] = step_multiplier*accur[i]
            end
        end
        converged = true
        for i in 1:n
            if abs(dx[i]) > accur[i]
                converged = false
                break
            end
        end
        if converged 
            print("convergence on position after ,$iteration, iterations")
            break
        end
    end
    return x, gradient, ihessian
end

