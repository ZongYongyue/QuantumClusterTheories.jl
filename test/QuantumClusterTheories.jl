using QuantumLattices
using ExactDiagonalization
using QuantumClusterTheories
using QuadGK
using Test
using StaticArrays: SVector
using QuantumClusterTheories: Perioder, Clusteration
using Printf: @printf, @sprintf
import QuantumLattices: update, update!
# function phase_by_azimuth(azs::AbstractVector, phs::AbstractVector)
#     function pbz(bond::Bond)
#         θ = azimuth(rcoordinate(bond))
#         for i in eachindex(azs)
#             any(≈(θ), azs[i]) && return phs[i]
#         end
#     end
#     return pbz
# end

# function ExactNormalGreenFunction(t, μ, k, ω; η=0.05)
#     ep = 2t*cos(k) + μ
#     g = 1/(ω - ep + η*im)
#     return [g 0 ; 0 g]
# end

# function ExactPxWaveGreenFunction(t, Δ, μ, k, ω; η=0.05)
#     ep, pm, pp = 2t*cos(k) + μ, -2im*sin(k), 2im*sin(k)
#     xi = √(ep^2 + (pm*pp)*Δ^2)
#     uv, u², v², g, l = abs(Δ)/(2*xi), (1+ep/xi)/2, (1-ep/xi)/2, 1/(ω - xi + η*im), 1/(ω + xi + η*im)
#     a = (g - l)
#     g11, g14, g32, g33  = u²*g + v²*l, uv*pm*a, uv*pp*a, v²*g + u²*l
#     g22, g23, g41, g44 = g11, g14, g32, g33
#     return [g11 0 0 g14; 0 g22 g23 0; 0 g32 g33 0; g41 0 0 g44]
# end

# unitcell = Lattice([0, 0]; vectors=[[1, 0]])
# cluster = Lattice(unitcell, (2,), ('p',))
# hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
# bs₁ = BinaryBases(1:4, 2)
# bs₂ = BinaryBases(1:2,3:4, 0.0)
# t = Hopping(:t, -1.0, 1) 
# p = Pairing(:p, 0.3, 1, (Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1))) + Coupling(Index(:, FID(1, -1//2, 1)), Index(:, FID(1, 1//2, 1))))/2; amplitude = phase_by_azimuth([(0, 2π), (π,)], [1.0, -1.0]))
# origiterms₁ = (t,)
# referterms₁ = (t,)
# origiterms₂ = (t, p)
# referterms₂ = (t, p)
# vca₁ = VCA(:N, unitcell, cluster, hilbert, origiterms₁, referterms₁, bs₁)
# vca₂ = VCA(:A, unitcell, cluster, hilbert, origiterms₂, referterms₂, bs₂)


# @testset "Perioder" begin
#     @test vca₁.perioder.map₁ == [[a, a+b] for a in [1, 3], b in [1]]
#     @test vca₁.perioder.map₂ == [[1, 2],]
# end

# @testset "singleParticleGreenFunction" begin
#     @test isapprox(singleParticleGreenFunction(:f, vca₁, [[-5π/12, 0.0],], [2.9,])[1][1], ExactNormalGreenFunction(-1.0, 0.0, -5π/12, 2.9); atol=1e-10)
#     @test isapprox(singleParticleGreenFunction(:f, vca₂, [[-5π/12, 0.0],], [2.9,])[1][1], ExactPxWaveGreenFunction(-1.0, 0.3, 0.0, -5π/12, 2.9); atol=1e-10)
# end

# @testset "GrandPotential" begin
#     rz = ReciprocalZone(reciprocals(cluster.vectors); length=10000)
#     gp₁ = GrandPotential(:f, vca₁, rz, 0.0)
#     gp₂ = GrandPotential(:f, vca₂, rz, 0.0)
#     t = 1.0
#     Δ = 0.3
#     μ = 0.0
#     ne = quadgk(k->-2t*cos(k) - μ , -π/2, π/2)[1]/π
#     pe = quadgk(k->-2t*cos(k) - μ - √((2t*cos(k) + μ)^2 + 4*((sin(k))^2)*Δ^2), 0, π)[1]/π
#     @test isapprox(gp₁, ne; atol=1e-10)
#     @test isapprox(gp₂, pe; atol=1e-10)
# end

mutable struct SVCA{U<:AbstractLattice, L<:AbstractLattice, C<:AbstractVector{L}, A<:AbstractLattice, G<:EDSolver, E<:AbstractVector{G}, P<:Perioder, T<:Partition, R<:AbstractVector{T}} <: Frontend
    const modelname::Union{String, Nothing}
    const cachepath::Union{String, Nothing}
    const unitcell::U
    const clusters::C
    const lattice::A
    const origigenerator::OperatorGenerator
    const refergenerator::OperatorGenerator
    solvers::E
    const perioder::P
    const partses::R
end

function SVCA(sym::Symbol, unitcell::AbstractLattice, clusters::AbstractVector{<:AbstractLattice}, clns::AbstractVector{<:Clusteration}, lvectors::AbstractVector, hilbert::Hilbert, origiterms::Tuple{Vararg{Term}}, referterms::Tuple{Vararg{Term}}; perioder::Union{Perioder, Nothing}=nothing, neighbors::Union{Nothing, Int, Neighbors}=nothing, modelname::Union{Nothing,String}=nothing, cachepath::Union{Nothing, String}=nothing)
    @assert length(clusters)==sum(length(cln.cid) for cln in clns) "The lengths of clusters and clusterations is inconsistent"
    table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
    lattice = Lattice(:lattice, hcat([cluster.coordinates for cluster in clusters]...), QuantumLattices.Spatials.vectorconvert(lvectors))
    splice = vcat([[i for _ in axes(clusters[i].coordinates, 2)] for i in eachindex(clusters)]...)
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, origiterms))
    origibonds = bonds(lattice, neighbors)
    referbonds = filter(bondd->isintracell(bondd), filter(bbond->isintraclusters(bbond, splice), origibonds))
    isnothing(perioder) ? (perioder = Perioder(unitcell, lattice, table)) : (perioder = perioder)
    origigenerator, refergenerator = OperatorGenerator(origiterms, origibonds, hilbert; table = table), OperatorGenerator(referterms, referbonds, hilbert; table = table)
    partses = Vector{Partition}(undef, length(clns))
    edsolvers = Vector{EDSolver}(undef, length(clns))
    for i in eachindex(clns)
        ctable = Table(clns[i].hilbert, Metric(EDKind(clns[i].hilbert), clns[i].hilbert))
        isnothing(clns[i].referterms) ? (creferterms=referterms) : (creferterms=clns[i].referterms)
        isnothing(clns[i].neighbors) ? (cneighbors=neighbors) : (cneighbors=clns[i].neighbors)
        partses[i] = Partition(sym, ctable, clns[i].bs)
        creferbonds = filter(bond -> isintracell(bond), bonds(clusters[clns[i].cid[1]], cneighbors))
        crefergenerator = OperatorGenerator(creferterms, creferbonds, clns[i].hilbert; table = ctable)
        if !isnothing(cachepath)
            modelname_str = isnothing(modelname) ? "default_model" : modelname
            cache_file_path = joinpath(cachepath, @sprintf "%s_%s_%s_%s.jls" modelname_str clns[i].cid repr(crefergenerator) EDSolver )
            if isfile(cache_file_path)
                edsolvers[i] = deserialize(cache_file_path)
                println("Load edsolver from $cache_file_path")
            else
                !isdir(cachepath)&& mkdir(cachepath)
                println("Cache directory created at $cachepath")
                edsolvers[i] = EDSolver(EDKind(clns[i].hilbert), partses[i], crefergenerator, clns[i].bs, ctable; m = clns[i].m)
                serialize(cache_file_path, edsolver)
                println("EDSolver cached at $cache_file_path")
            end
        else
            edsolvers[i] = EDSolver(EDKind(clns[i].hilbert), partses[i], crefergenerator, clns[i].bs, ctable; m = clns[i].m)
        end
    end
    return SVCA(modelname, cachepath, unitcell, clusters, lattice, origigenerator, refergenerator, edsolvers, perioder, partses)
end
isintraclusters(bond::Bond, splice::AbstractVector{<:Integer}) = all(point-> splice[bond.points[1].site]==splice[point.site] , bond)

function update!(svca::SVCA, oparams::Parameters, rparams::Parameters, clns::AbstractVector{<:Clusteration})
    update!(svca.origigenerator; oparams...)
    update!(svca.refergenerator; rparams...)
    cachepath = svca.cachepath
    modelname = svca.modelname
    edsolvers = Vector{EDSolver}(undef, length(clns))
    for i in eachindex(clns)
        ctable = Table(clns[i].hilbert, Metric(EDKind(clns[i].hilbert), clns[i].hilbert))
        creferterms=clns[i].referterms
        isnothing(clns[i].neighbors) ? (cneighbors = maximum(term->term.bondkind, creferterms)) : (cneighbors=clns[i].neighbors)
        creferbonds = filter(bond -> isintracell(bond), bonds(svca.clusters[clns[i].cid[1]], cneighbors))
        crefergenerator = OperatorGenerator(creferterms, creferbonds, clns[i].hilbert; table = ctable)
        if !isnothing(cachepath)
            modelname_str = isnothing(modelname) ? "default_model" : modelname
            cache_file_path = joinpath(cachepath, @sprintf "%s_%s_%s_%s.jls" modelname_str clns[i].cid repr(crefergenerator) EDSolver)
            if isfile(cache_file_path)
                edsolvers[i] = deserialize(cache_file_path)
                println("Load edsolver from $cache_file_path")
            else
                !isdir(cachepath)&& mkdir(cachepath)
                println("Cache directory created at $cachepath")
                edsolvers[i] = EDSolver(EDKind(clns[i].hilbert), svca.partses[i], crefergenerator, svca.partses[i].sector, ctable; m = clns[i].m)
                serialize(cache_file_path, edsolver)
                println("EDSolver cached at $cache_file_path")
            end
        else
            edsolvers[i] = EDSolver(EDKind(clns[i].hilbert), svca.partses[i], crefergenerator, svca.partses[i].sector, ctable; m = clns[i].m)
        end
    end
    svca.solvers = edsolvers
    return svca
end



unitcell = Lattice([0,0]; vectors=[[1,0],[0,1]])
clusters = [Lattice([0,0],[0,1]), Lattice([1,0],[1,1])]
# lattice = Lattice([0,0],[0,1],[1,0],[1,1];vectors=[[2,0],[0,2]])
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:4)
bs= Sector(hilbert, ParticleNumber(4))
t = Hopping(:t, -1.0, 1)
origiterms = (t,)
referterms = (t,)
# table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
# neighbors = maximum(term->term.bondkind, origiterms)
# origibonds = bonds(lattice, neighbors)
# referbonds = filter(bond -> isintracell(bond), origibonds)
# origigenerator, refergenerator = OperatorGenerator(origiterms, origibonds, hilbert; table = table), OperatorGenerator(referterms, referbonds, hilbert; table = table)
# perioder = QuantumClusterTheories.Perioder(unitcell, lattice, table) 
# partses = [Partition(:N, table, bs)]
# edsolvers = [EDSolver(EDKind(hilbert), partses[1], refergenerator, bs, table; m = 200)]
# svca = SVCA(nothing, nothing,unitcell, clusters, lattice, [0,0],origigenerator, refergenerator,edsolvers, perioder, partses)

clns = [Clusteration(Hilbert(site=>Fock{:f}(1, 2) for site=1:length(clusters[i])), BinaryBases(4,2); cid=[i]) for i in eachindex(clusters)]

svca = SVCA(:N,unitcell, clusters, clns, [[2,0],[0,2]], hilbert, origiterms, referterms)

clns2 = [Clusteration(Hilbert(site=>Fock{:f}(1, 2) for site=1:length(clusters[i])), BinaryBases(4,2); cid=[i], referterms=(Hopping(:t, -2.0, 1),)) for i in eachindex(clusters)]

update!(svca, (t=-2,), (t=-3,),  clns2)
# function VCAS(sym::Symbol, unitcell::AbstractLattice, clusters::AbstractVector{<:AbstractLattice}, hilberts::AbstractVector{<:Hilbert}, origiterms::Tuple{Vararg{Term}}, referterms::Tuple{Vararg{Term}}, bs::BinaryBases; perioder::Union{Perioder, Nothing}=nothing, neighbors::Union{Nothing, Int, Neighbors}=nothing, m::Int=200, modelname::Union{Nothing,String}=nothing, cachepath::Union{Nothing, String}=nothing)
#     table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
#     isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, origiterms))
#     origibonds = bonds(cluster, neighbors)
#     referbonds = filter(bond -> isintracell(bond), origibonds)
#     isnothing(perioder) ? (perioder = Perioder(unitcell, cluster, table)) : (perioder = perioder)
#     parts = Partition(sym, table, bs)
#     origigenerator, refergenerator = OperatorGenerator(origiterms, origibonds, hilbert; table = table), OperatorGenerator(referterms, referbonds, hilbert; table = table)
#     if !isnothing(cachepath)
#         modelname_str = isnothing(modelname) ? "default_model" : modelname
#         cache_file_path = joinpath(cachepath, @sprintf "%s_%s_%s_%s.jls" modelname_str cluster.name repr(refergenerator) EDSolver )
#         if isfile(cache_file_path)
#             edsolver = deserialize(cache_file_path)
#             println("Load edsolver from $cache_file_path")
#         else
#             !isdir(cachepath)&& mkdir(cachepath)
#             println("Cache directory created at $cachepath")
#             edsolver = EDSolver(EDKind(hilbert), parts, refergenerator, bs, table; m = m)
#             serialize(cache_file_path, edsolver)
#             println("EDSolver cached at $cache_file_path")
#         end
#     else
#         edsolver = EDSolver(EDKind(hilbert), parts, refergenerator, bs, table; m = m)
#     end
#     return VCA(modelname, cachepath, unitcell, cluster, origigenerator, refergenerator, edsolver, perioder, parts)
# end

# struct Lattice{N, D<:Number, M} <: AbstractLattice{N, D, M}
#     name::Symbol
#     coordinates::Matrix{D}
#     vectors::SVector{M, SVector{N, D}}
#     function Lattice(name::Symbol, coordinates::AbstractMatrix{<:Number}, vectors::SVector{M, <:SVector{N, <:Number}}) where {N, M}
#         @assert N==size(coordinates, 1) "Lattice error: shape mismatched."
#         datatype = promote_type(Float, eltype(coordinates), eltype(eltype(vectors)))
#         coordinates = convert(Matrix{datatype}, coordinates)
#         vectors = convert(SVector{M, SVector{N, datatype}}, vectors)
#         new{N, datatype, M}(name, coordinates, vectors)
#     end
# end
