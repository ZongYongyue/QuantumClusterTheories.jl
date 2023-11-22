using QuantumLattices
using QuantumClusterTheories
using ExactDiagonalization
using LinearAlgebra
using Serialization
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
