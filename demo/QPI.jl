using QuantumLattices
using Plots
colorbar = cgrad(:linear_tritanopic_krjcw_5_98_c46_n256, rev = true);
using ExactDiagonalization
using QuantumClusterTheories



function delta_scattering_QPI(sym::Symbol, vca::VCA, k_path::Union{AbstractVector, ReciprocalSpace}, ω::Number, V::Number; μ::Real=0.0, η::Real=0.05, loc::Union{Nothing, AbstractVector}=nothing)
    ω = ω + μ + η*im
    oops, rops = filter(op -> length(op) == 2, collect(expand(vca.origigenerator))), filter(op -> length(op) == 2, collect(expand(vca.refergenerator)))
    R, N = isempty(filter(op -> op.id[1].index.iid.nambu==op.id[2].index.iid.nambu, collect(rops))), length(vca.refergenerator.table)
    R ? N=N : N=2*N
    oopsseqs = QuantumClusterTheories.seqs(oops, vca.origigenerator.table)
    rm = QuantumClusterTheories.referQuadraticTerms(R, rops, zeros(ComplexF64, N, N), vca.refergenerator.table)
    gfpv = QuantumClusterTheories.GreenFunctionPath(R, zeros(ComplexF64, N, N),  oops, oopsseqs, rm, vca.perioder, vca.cluster, k_path, ClusterGreenFunction(R, sym, vca.solver, ω); loc=loc)
    ivm = Matrix{ComplexF64}(I, size(gfpv[1])...)/V
    gvs = [gf*inv(ivm-gf) for gf in gfpv]
    rs = zeros(Float64, length(gfpv))
    ms = Momenta(Momentum₂{Int(√(length(k_path))), Int(√(length(k_path)))})
    for q in eachindex(gfpv) 
        for k in eachindex(gfpv)
            rs[q] += (-1/π)*tr(gvs[k]*gfpv[findfirst(ms[k]+ms[q],ms)]).im
        end
    end
    return rs
end

const δ_QPI = delta_scattering_QPI





unitcell = Lattice([0, 0]; vectors=[[1, 0], [0, 1]])
cluster = Lattice(unitcell, (2,2), ('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = Sector(hilbert, SpinfulParticle(4, 0.0))
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 4.0)
μ = 1
origiterms = (t, U)
referterms = (t, U)
neighbors = Neighbors(0=>0.0, 1=>1.0)
@time vca = VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs)
k_path = ReciprocalPath(reciprocals(unitcell.vectors), rectangle"Γ-X-M-Γ", length=100)
dz = BrillouinZone(reciprocals(unitcell.vectors), 100)

@time qpi =  δ_QPI(:f, vca, dz, 0, 0.05; μ=μ, η=0.05)

f = heatmap(range(-2pi,2pi,100),range(-2pi,2pi,100),reshape(qpi, Int(√(length(dz))), Int(√(length(dz)))),ratio=1)

gfv = singleParticleGreenFunction(:f, vca, dz, 0:1:0; μ=μ, η=0.05)

s = heatmap(range(-2pi,2pi,100),range(-2pi,2pi,100),reshape(spectrum(gfv), Int(√(length(dz))), Int(√(length(dz)))), ratio=1, color=colorbar,clims=(0, 3))

sf = plot(s, f, grid=false,xaxis=false, yaxis=false,colorbar=false,size=(800,400))