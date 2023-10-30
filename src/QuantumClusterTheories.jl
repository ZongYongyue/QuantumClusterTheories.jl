module QuantumClusterTheories

using ExactDiagonalization
using LinearAlgebra
using QuadGK
using QuantumLattices
using Distributed
using Serialization
using TimerOutputs

export VCA, singleParticleGreenFunction, clusterspectrum, spectrum, GrandPotential, Optimal, optimals, OrderParameters
export antiferro, belongs, moirephase, spinrotation, dx2y2, es, dxy, saveData, loadData, spawn

#colorbar = cgrad(:linear_tritanopic_krjcw_5_98_c46_n256, rev = true)
const vcatimer = TimerOutput()

struct Perioder{P<:AbstractVector{<:Integer}, T<:AbstractArray{P}, S<:AbstractArray{P}, C<:AbstractVector{<:Tuple}}
    map₁::T
    map₂::S
    channels::C
end 
function Perioder(unitcell::AbstractLattice, cluster::AbstractLattice, table::Table)
    @assert !isempty(unitcell.vectors) "the vectors in unitcell cannot be empty !"
    seq = sort(collect(keys(table)), by = x -> table[x])
    nspin, norbi, channels = sort(collect(Set(key[1] for key in seq))), sort(collect(Set(key[3] for key in seq))), Vector{Tuple{Vector{Int64},Vector{Int64}}}()
    map₁ = [filter(k -> nspin[i]==seq[k][1]&&norbi[j]==seq[k][3], 1:length(seq)) for i in eachindex(nspin), j in eachindex(norbi)]
    map₂ = [filter(j -> issubordinate(cluster.coordinates[:,j]-unitcell.coordinates[:,i], unitcell.vectors), 1:length(cluster)) for i in 1:length(unitcell)]
    for i in 1:size(map₁, 1), j in 1:size(map₁, 2), u in 1:size(map₁, 1), v in 1:size(map₁, 2) 
        push!(channels, ([i, j], [u, v]))
    end
    return Perioder(map₁, map₂, channels)
end

struct VCA{L<:AbstractLattice, G<:EDSolver, P<:Perioder} <: Frontend
    unitcell::L
    cluster::L
    origigenerator::OperatorGenerator
    refergenerator::OperatorGenerator
    solver::G
    perioder::P
end

function VCA(sym::Symbol, unitcell::AbstractLattice, cluster::AbstractLattice, hilbert::Hilbert, origiterms::Tuple{Vararg{Term}}, referterms::Tuple{Vararg{Term}}, bs::BinaryBases, rparam::Union{Parameters, Nothing}=nothing;neighbors::Union{Nothing, Int, Neighbors}=nothing, m::Int=200)
    table = Table(hilbert, Metric(EDKind(typeof(origiterms)), hilbert))
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, origiterms))
    origibonds = bonds(cluster, neighbors)
    referbonds = filter(bond -> isintracell(bond), origibonds)
    origigenerator, refergenerator = OperatorGenerator(origiterms, origibonds, hilbert; table = table), OperatorGenerator(referterms, referbonds, hilbert; table = table)
    isnothing(rparam) || update!(refergenerator; rparam...)
    edsolver = EDSolver(EDKind(typeof(origiterms)), sym, refergenerator, bs, table; m = m)
    perioder = Perioder(unitcell, cluster, table)         
    return VCA(unitcell, cluster, origigenerator, refergenerator, edsolver, perioder)
end

function origiQuadraticTerms!(normal::Bool, om::AbstractMatrix, oops::AbstractVector, oopsseqs::AbstractVector,k::AbstractVector)
    pm = copy(om)
    for i in eachindex(oops)
        seq₁, seq₂, n₁, n₂ = oopsseqs[i][1], oopsseqs[i][2], oopsseqs[i][3], oopsseqs[i][4]
        phase₁, phase₂ = exp(im*dot(k, icoordinate(oops[i]))), exp(-im*dot(k, icoordinate(oops[i])))
        n₁==2&&n₂==1 ? (om[seq₁, seq₂]+=oops[i].value*phase₁) : (n₁==2&&n₂==2 ? (om[seq₁, seq₂]+=oops[i].value*phase₁) : (om[seq₁, seq₂]+=oops[i].value*phase₂))
        (n₁==2&&n₂==1) ? (pm[seq₁, seq₂] += oops[i].value*phase₂) : pm[seq₁, seq₂]=pm[seq₁, seq₂]
    end
    if !normal
        om[Vector((size(om,1)÷2+1):size(om,1)), Vector((size(om,1)÷2+1):size(om,1))] = -transpose(pm[Vector(1:(size(om,1)÷2)), Vector(1:(size(om,1)÷2))])
    end
    return om
end

function referQuadraticTerms(normal::Bool, rops::AbstractVector, rm::AbstractMatrix, table::Table)
    for rop in rops 
        rop[1].index.iid.nambu==2 ? seq₁=table[rop[1].index'] : seq₁=(table[rop[1].index]+length(table))
        rop[2].index.iid.nambu==1 ? seq₂=table[rop[2].index] : seq₂=(table[rop[2].index']+length(table))
        rop[1].index.iid.nambu==2&&rop[2].index.iid.nambu==1 ? (rm[seq₁, seq₂] += rop.value) : (rop[2].index.iid.nambu==2 ? (rm[seq₁, seq₂] += rop.value) : (rm[seq₁, seq₂] += rop.value))
    end
    if !normal
        rm[Vector(length(table)+1:2*length(table)),Vector(length(table)+1:2*length(table))] = -transpose(rm[Vector(1:length(table)),Vector(1:length(table))])
    end
    return rm
end

function seqs(oops::AbstractVector, table::Table)
    seqs = [zeros(Int, 2) for _ in 1:length(oops)]
    for i in eachindex(oops) 
        oops[i][1].index.iid.nambu==2 ? seq₁=table[oops[i][1].index'] : seq₁=(table[oops[i][1].index]+length(table))
        oops[i][2].index.iid.nambu==1 ? seq₂=table[oops[i][2].index] : seq₂=(table[oops[i][2].index']+length(table))
        seqs[i] = [seq₁, seq₂, oops[i][1].index.iid.nambu, oops[i][2].index.iid.nambu]
    end
    return seqs
end

function CPTcore(cgfm::AbstractMatrix, vm::AbstractMatrix)
    result = Matrix{ComplexF64}(I, size(vm)...)
    return cgfm*inv(mul!(result, vm, cgfm, -1, 1))
end

function periodization(coeff::Tuple, gfml::AbstractMatrix, map₂::AbstractVector, coordinates::AbstractMatrix, k::AbstractVector)
    (x, y), N, L = coeff, length(map₂), size(coordinates,2)
    lgfm, pgfm =Matrix{ComplexF64}(undef, L, L), Matrix{ComplexF64}(undef, N, N)
    for i in 1:L, j in 1:L
        @views ra, rb = coordinates[:, i], coordinates[:, j]
        lgfm[i, j] = gfml[i,j]*exp(-im*dot(k, (x*ra - y*rb)))
    end
    for m in 1:N, n in 1:N
        @views cmap₂, hmap₂ = map₂[m], map₂[n]
        pgfm[m, n] = (N/L)*sum(lgfm[cmap₂, hmap₂])
    end
    return pgfm
end

function perGreenFunction(coeff::Tuple, GFm::AbstractMatrix, k::AbstractVector, perioder::Perioder, cluster::AbstractLattice)
    gfv = Vector{Matrix{ComplexF64}}(undef, length(perioder.channels))
    gfm = Matrix{ComplexF64}(undef, length(perioder.map₁)*length(perioder.map₂), length(perioder.map₁)*length(perioder.map₂))
    for i in eachindex(perioder.channels)
        gfml = GFm[perioder.map₁[perioder.channels[i][1]...], perioder.map₁[perioder.channels[i][2]...]]
        gfv[i] = periodization(coeff, gfml, perioder.map₂, cluster.coordinates, k)
    end
    gfmm = reshape(gfv, (length(perioder.map₁), length(perioder.map₁)))
    for i in 1:length(perioder.map₁), j in 1:length(perioder.map₁)
        for u in 1:length(perioder.map₂), v in 1:length(perioder.map₂)
            gfm[(i-1)*length(perioder.map₂) + u, (j-1)*length(perioder.map₂) + v] = gfmm[i, j][u, v]
        end
    end
    return gfm
end

function GreenFunctionPath(normal::Bool, om::AbstractMatrix, oops::AbstractVector, oopsseqs::AbstractVector, rm::AbstractMatrix, perioder::Perioder, cluster::AbstractLattice, k_path::ReciprocalSpace, CGFm::AbstractMatrix)
    gfpath = Vector{Matrix{ComplexF64}}(undef, length(k_path))
    if normal
        for i in eachindex(k_path)
            dest = copy(om)
            Vm = origiQuadraticTerms!(normal, dest, oops, oopsseqs, k_path[i]) - rm
            gfpath[i] = perGreenFunction((1, 1), CPTcore(CGFm, Vm), k_path[i], perioder, cluster)
        end
    else
        #coeffs, arrs = [(1, 1), (1, -1), (-1, 1), (-1, -1)], [[1:(size(om,1)÷2), 1:(size(om,1)÷2)], [1:(size(om,1)÷2), ((size(om,1)÷2)+1):(size(om,1))], [((size(om,1)÷2)+1):(size(om,1)), 1:(size(om,1)÷2)], [((size(om,1)÷2)+1):(size(om,1)), ((size(om,1)÷2)+1):(size(om,1))]]
        for i in eachindex(k_path)
            dest = copy(om)
            Vm = origiQuadraticTerms!(normal, dest, oops, oopsseqs, k_path[i]) - rm
            GFm = CPTcore(CGFm, Vm)
            #a, b, c, d = [perGreenFunction(coeffs[i], GFm[arrs[i]...], k_path[i], perioder, cluster) for i in 1:4]
            gfpath[i] = GFm#[a b; c d]
        end
    end
    return gfpath
end 

function singleParticleGreenFunction(sym::Symbol, vca::VCA, k_path::ReciprocalSpace, ω_range::AbstractRange)
    oops, rops = filter(op -> length(op) == 2, collect(expand(vca.origigenerator))), filter(op -> length(op) == 2, collect(expand(vca.refergenerator)))
    R, N = isempty(filter(op -> op.id[1].index.iid.nambu==op.id[2].index.iid.nambu, collect(rops))), length(vca.refergenerator.table)
    R ? N=N : N=2*N
    oopsseqs = seqs(oops, vca.origigenerator.table)
    rm = referQuadraticTerms(R, rops, zeros(ComplexF64, N, N), vca.refergenerator.table)
    gfpv = [GreenFunctionPath(R, zeros(ComplexF64, N, N), oops, oopsseqs, rm, vca.perioder, vca.cluster, k_path, ClusterGreenFunction(R, sym, vca.solver, ω)) for ω in ω_range]
    return gfpv
end

function clusterspectrum(cgfpathv::AbstractVector)
    A = zeros(Float64, length(cgfpathv))
    for j in eachindex(cgfpathv)
        A[j] = (-1/π) * (tr(cgfpathv[j])).im
    end
    return A
end
function spectrum(gfpathv::AbstractVector)
    A = zeros(Float64, length(gfpathv), length(gfpathv[1]))
        for i in eachindex(gfpathv)
            for j in eachindex(gfpathv[i])
                A[i, j] = (-1/π) * (tr(gfpathv[i][j])).im
            end
        end
        return A
end


function GPcore(temp::AbstractMatrix, cgfm::AbstractMatrix, vm::AbstractMatrix)
    result = copy(temp)
    mul!(result, vm, cgfm, -1, 1)
    return log(abs(det(result)))
end

function GPintegrand(normal::Bool, sym::Symbol, solver::EDSolver, temp::AbstractMatrix, vmvec::AbstractVector, ω::Complex)
    intra = 0.0
    cgfm = ClusterGreenFunction(normal, sym, solver, ω)
    for i in eachindex(vmvec)          
        intra += GPcore(temp, cgfm, vmvec[i])
    end
    return intra
end 

function GrandPotential(sym::Symbol, vca::VCA, bz::AbstractVector, μ::Real) 
    oops, rops = filter(op -> length(op) == 2, collect(expand(vca.origigenerator))), filter(op -> length(op) == 2, collect(expand(vca.refergenerator)))
    R, N = isempty(filter(op -> op.id[1].index.iid.nambu==op.id[2].index.iid.nambu, collect(rops))), length(vca.refergenerator.table)
    R ? N=N : N=2*N
    oopsseqs = seqs(oops, vca.origigenerator.table)
    rm = referQuadraticTerms(R, rops, zeros(ComplexF64, N, N), vca.refergenerator.table)
    temp = Matrix{ComplexF64}(I, N, N)
    vmvec = Vector{Matrix{ComplexF64}}(undef,length(bz))
    for i in eachindex(bz)
        vmvec[i] = origiQuadraticTerms!(R, zeros(ComplexF64, N, N), oops, oopsseqs, bz[i]) - rm
    end
    trvm = sum([tr(vmv) for vmv in vmvec])
    gp = (vca.solver.gse + (1/length(bz))*(- quadgk(x -> GPintegrand(R, sym, vca.solver, temp, vmvec, x*im+μ), 0, Inf)[1]/π + trvm.re/2))/length(vca.cluster)/length(vca.unitcell)
    return gp
end

function antiferro(wavevector::AbstractVector)
    function bondfunction(bond::Bond)
        return exp(im*dot(wavevector,rcoordinate(bond)))
    end
    return bondfunction
end

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

function moirephase(ϕ::Real, spin::Rational)
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

function dx2y2(bond::Bond)
    θ = azimuth(rcoordinate(bond))
    any(≈(θ),(0, π, 2π)) && return 1.0
    any(≈(θ),(π/2, 3π/2)) && return -1.0
end

function dxy(bond::Bond)
    θ = azimuth(rcoordinate(bond))
    any(≈(θ),(π/4, 5π/4)) && return 1.0
    any(≈(θ),(3π/4, 7π/4)) && return -1.0
end

function spinrotation(θ::Real, ϕ::Real)
    sin(θ)*cos(ϕ)*MatrixCoupling(:, FID, :, σ"x", :) + sin(θ)*sin(ϕ)*MatrixCoupling(:, FID, :, σ"y", :) + cos(θ)*MatrixCoupling(:, FID, :, σ"z", :)
end

macro setup_worker()
    quote
        using QuantumClusterTheories
    end
end

function spawn(numworkers::Int)
    np = length(workers())
    np == 1 ? addworkers=numworkers : (np < numworkers ? addworkers=(numworkers - np) : addworkers = 0)
    addprocs(addworkers)
    @everywhere collect(2:numworkers+1) include("/Users/tzung/Library/Mobile Documents/com~apple~CloudDocs/Clone/Bro-Gu/QuantumClusterTheories.jl/src/parallelusing.jl")
end

struct Optimal{T<:VCA, P<:Parameters}
    optvca::T
    optparams::P
    function Optimal(optvca::VCA, optparams::Parameters)
        new{typeof(optvca),typeof(optparams)}(optvca, optparams)
    end
end

function optimals(vcas::AbstractArray{<:VCA}, gps::AbstractArray{<:Real},varparams::AbstractArray{<:Parameters})
    if ndims(varparams)==2
        opts = Vector{Optimal}(undef, size(varparams, 1))
        for i in axes(varparams,1)
            Δgps = gps[i,:] .- maximum(gps[i,:])
            index = argmin(Δgps)
            opts[i] = Optimal(vcas[i, index], varparams[i, index])
        end
        return opts
    end
end

function OPintegrand(sym::Symbol, vca::VCA, bz::ReciprocalSpace, iω::Complex, sm::AbstractMatrix, oops::AbstractVector, oopsseqs::AbstractVector, rm::AbstractMatrix, μ::Real)
    om = zeros(ComplexF64, length(vca.refergenerator.table), length(vca.refergenerator.table))
    cgfm = ClusterGreenFunction(sym, vca.solver, iω+μ)
    intra = 0.0
    for k in bz
        vm = origiQuadraticTerms!(copy(om), oops, oopsseqs, k) - rm
        gfm = CPTcore(cgfm, vm)
        intra += (tr(sm*gfm) - tr(sm)/(iω-1.0)).re
    end
    return intra
end

function OrderParameters(opt::Optimal, hilbert::Hilbert, bz::ReciprocalSpace, term::Term, μ::Real)
    vca = opt.optvca
    term.value = convert(typeof(term.value), 1.0)
    sm = referQuadraticTerms(collect(expand(term, filter(bond -> isintracell(bond), bonds(vca.cluster, term.bondkind)), hilbert)), vca.refergenerator.table)
    oops = filter(op -> length(op) == 2, collect(expand(vca.origigenerator)))
    oopsseqs = seqs(oops, vca.origigenerator.table)
    rm = referQuadraticTerms(filter(op -> length(op) == 2, collect(expand(vca.refergenerator))), vca.refergenerator.table)
    return abs((1/length(bz))*quadgk(x -> OPintegrand(vca, bz, x*im, sm, oops, oopsseqs, rm, μ), 0, Inf)[1]/π/length(vca.unitcell)/length(vca.cluster))
end

function saveData(data, filename::String)
    open(filename, "w") do io
        serialize(io, data)
    end
end

function loadData(filename::String)
    data = nothing
    open(filename, "r") do io
        data = deserialize(io)
    end
    return data
end


end #module
