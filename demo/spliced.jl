using QuantumLattices
using Plots
colorbar = cgrad(:linear_tritanopic_krjcw_5_98_c46_n256, rev = true);
using ExactDiagonalization
using QuantumClusterTheories


unitcell = Lattice([0,0],[√3/2,-1/2],[√3/2, 3/2], [0, 1]; vectors=[[√3,0], [0,3]])
punitcell = Lattice(:per, (Lattice(unitcell, (1,3), ('p','p'))).coordinates, QuantumLattices.Spatials.vectorconvert([[√3,0], [0,1000]]))
cluster = Lattice(unitcell, (2,1), ('p','p'))
clusters = [Lattice(cluster, (0:0,(i-1):(i-1)), ('p','p')) for i in 1:3]

hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:sum(length(cluster) for cluster in clusters))

t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 2.9)
origiterms = (t, U)
referterms = (t, U)
table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
perioder = QuantumClusterTheories.Perioder(punitcell, Lattice(:lattice, hcat([cluster.coordinates for cluster in clusters]...), QuantumLattices.Spatials.vectorconvert([[2√3,0]])), table)
clns = [Clusteration(Hilbert(site=>Fock{:f}(1, 2) for site=1:length(clusters[1])), BinaryBases(16,8); cid=Vector(1:3))]
svca = SVCA(:N, punitcell, clusters, clns, [[2√3,0]], hilbert, origiterms, referterms; perioder=perioder)

k_path = ReciprocalPath(reciprocals([[√3,0]]), line"Γ₁-Γ₂", length=100)
ω_range = range(-4, 4, length=300)
@time G = spliceLatticeGreenFunction(:f, svca, k_path, ω_range; μ=U.value/2)
A = spectrum(G)
f = plot(k_path, ω_range, A; xlabel="k", ylabel="ω", color=colorbar, title="Spectral Function",clims=(0, 10))

