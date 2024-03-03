include("../src/tools.jl")
import QuantumClusterTheories: seqs, referQuadraticTerms, origiQuadraticTerms!, CPTcore
using QuadGK
using Plots
function Disintegrand(sym::Symbol, vca::VCA, k::AbstractVector, iω::Complex, μ::Real, select::Union{Nothing, AbstractVector})
    gfm = singleParticleGreenFunction(sym, vca, [k,], [iω,]; η=0.0, μ=μ)[1][1]
    select==nothing ? gfm=gfm : gfm=gfm[select, select]
    intra = (tr(gfm) - size(gfm,1)/(iω-1.0)).re
    return intra
end

function DistributionFunction(sym::Symbol, vca::VCA, bz::ReciprocalSpace, μ::Real; select::Union{Nothing, AbstractVector}=nothing)
    ds = zeros(Float64, length(bz))
    for i in eachindex(bz)
        ds[i] = quadgk(x -> Disintegrand(sym, vca, bz[i], x*im, μ, select), 0, Inf)[1]/π
    end
    return ds
end

unitcell = Lattice([0, 0]; vectors=[[1, 0]])
cluster = Lattice(unitcell, (2,), ('p',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs= Sector(hilbert, ParticleNumber(2))
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 2)
origiterms = (t, U)
referterms = (t, U)
vca = VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs)
k_path = ReciprocalPath(reciprocals(unitcell.vectors), line"Γ-X", length=100)

dstr = DistributionFunction(:f, vca, k_path, 1; select=[1,])
f = plot(k_path, dstr)

#=
unitcell = Lattice([0, 0]; vectors = [[√3/2, 1/2], [0, 1]])
cluster = Lattice(unitcell, (2,2),('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = Sector(hilbert, ParticleNumber(4))

t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 3.0)
origiterms = (t, U)
referterms = (t, U)
vca = VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs)
k_path = ReciprocalPath(reciprocals(unitcell.vectors), hexagon"Γ-K, 120°", length=100)

dstr = Dstr(:f, vca, k_path, 1.5)
f = plot(k_path, dstr)
=#