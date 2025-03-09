using QuantumLattices
using QuantumClusterTheories
using ExactDiagonalization
using Plots
colorbar = cgrad(:linear_tritanopic_krjcw_5_98_c46_n256, rev = true);

unitcell = Lattice([0, 0]; vectors=[[1, 0]])
cluster = Lattice(unitcell, (10,), ('p',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs= Sector(hilbert, ParticleNumber(10))
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 8.0)
origiterms = (t, U)
referterms = (t, U)
@time vca = VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs)
k_path = ReciprocalPath(reciprocals(unitcell.vectors), line"X₂-Γ-X", length=200)
ω_range = range(-4, 12, length=300)
@time G = singleParticleGreenFunction(:f, vca, k_path, ω_range; μ=0.0)
A = spectrum(G)
f = plot(k_path, ω_range, A; xlabel="k", ylabel="ω", color=colorbar , title="Spectral Function",clims=(0, 3))