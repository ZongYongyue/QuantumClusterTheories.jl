using QuantumLattices
using ExactDiagonalization
using QuantumClusterTheories
using Plots
using Distributed


colorbar = cgrad(:linear_tritanopic_krjcw_5_98_c46_n256, rev = true)
#=
unitcell = Lattice([0, 0]; vectors=[[1, 0]])
cluster = Lattice(unitcell, (4,), ('p',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs= BinaryBases(1:8, 4)⊗BinaryBases(9:16, 4)
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 2.0)
origiterms = (t, U)
t_r = Hopping(:t, -1.0, 1)
referterms = (t_r, U)
neighbors = Neighbors(0=>0.0, 1=>1.0)
@time vca = VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs)
k_path = ReciprocalPath(reciprocals(unitcell.vectors), line"X₂-Γ-X", length=300)
ω_range = range(-4, 4, length=400)
fq = ω_range .+ (Parameters(vca.refergenerator)[:U]/2 + 0.05*im)
@time G = singleParticleGreenFunction(:f, vca, k_path, fq)
A = spectrum(G)
f = plot(k_path, ω_range, A; xlabel="k", ylabel="ω", color=colorbar , title="Spectral Function",clims=(0, 3))
=#

unitcell = Lattice([0, 0]; vectors=[[1, 0], [0, 1]])
cluster = Lattice(unitcell, (2,2), ('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = BinaryBases(1:4, 2)⊗BinaryBases(5:8, 2)
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 4.0)
origiterms = (t, U)
referterms = (t, U)
neighbors = Neighbors(0=>0.0, 1=>1.0)
@time vca = VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs)
k_path = ReciprocalPath(reciprocals(unitcell.vectors), rectangle"Γ-X-M-Γ", length=100)
ω_range = range(-6, 6, length=400)
fq = ω_range .+ (Parameters(vca.refergenerator)[:U]/2 + 0.05*im)
@time G = singleParticleGreenFunction(:f, vca, k_path, fq)
A = spectrum(G)
f = plot(k_path, ω_range, A; xlabel="k", ylabel="ω", color=colorbar, title="Spectral Function",clims=(0, 3))

#=
unitcell = Lattice([0, 0]; vectors=[[1, 0],[0, 1]])
cluster = Lattice(unitcell,(2,2),('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = BinaryBases(1:4, 2)⊗BinaryBases(5:8, 2)
t = Hopping(:t, Complex(-1.0), 1)
U = Hubbard(:U, Complex(0.0))
μ = Onsite(:μ, Complex(-0.0))
origiterms = (t, U, μ)
t_r = Hopping(:t, Complex(-1.0), 1)
af = Onsite(:af, Complex(0.0), MatrixCoupling(:, FID, :, σ"z", :); amplitude=antiferro([π, π]))
referterms = (t_r, U, μ, af)

varparams = [(U = u, af = a) for u in [2,4,8,12,16], a in range(0, 0.3, 50)]
neighbors = Neighbors(0=>0.0, 1=>1.0)
rz = ReciprocalZone(reciprocals(cluster.vectors); length=100)

spawn(8)

@time vcas = pmap(param -> VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs, param; neighbors=neighbors, m=200), varparams)
@time gps = pmap(vca -> GrandPotential(:f, vca, rz, real(Parameters(vca.refergenerator)[:U]/2)), vcas)
#20.862047 seconds (6.14 M allocations: 454.191 MiB, 0.38% gc time, 4.16% compilation time)
#65.570335 seconds (3.84 M allocations: 255.548 MiB, 0.09% gc time, 0.92% compilation time)
=#

