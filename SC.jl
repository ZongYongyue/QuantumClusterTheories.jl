using QuantumLattices
using ExactDiagonalization
using QuantumClusterTheories
using Plots
using Distributed

unitcell = Lattice([0, 0]; vectors=[[1, 0], [0, 1]])
cluster = Lattice(unitcell, (2,2), ('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = BinaryBases([1:4,5:8], 0.0)
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 8.0)
μ = Onsite(:μ, 1.2)
coupling1=Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1)))
coupling2=Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1)))-Coupling(Index(:, FID(1, -1//2, 1)), Index(:, FID(1, 1//2, 1)))
origiterms = (t, U, μ)
neighbors = Neighbors(0=>0.0, 1=>1.0, 2=>√2)

s = Pairing(:s, 0.3, 0, coupling1)
referterms1 = (t, U, μ, s)

D = Pairing(:D, 0.3, 1, coupling2; amplitude=dx2y2)
referterms2 = (t, U, μ, D)

Es = Pairing(:Es, 0.3, 1, coupling2)
referterms3 = (t, U, μ, Es)

Dxy = Pairing(:Dxy, 0.3, 2, coupling2; amplitude=dxy)
referterms4 = (t, U, μ, Dxy)

varparams1 = [(s = a,) for a in range(1e-9, 0.2, 20)]
varparams2 = [(D = a,) for a in range(1e-9, 0.2, 20)]
varparams3 = [(Es = a,) for a in range(1e-9, 0.2, 20)]
varparams4 = [(Dxy = a,) for a in range(1e-9, 0.2, 20)]
rz = ReciprocalZone(reciprocals(cluster.vectors); length=100)

spawn(8)

@time vcas1 = pmap(param -> VCA(:A, unitcell, cluster, hilbert, origiterms, referterms1, bs, param; neighbors=neighbors, m=200), varparams1)
@time gps1 = pmap(vca -> GrandPotential(:f, vca, rz, 0.0), vcas1)

@time vcas2 = pmap(param -> VCA(:A, unitcell, cluster, hilbert, origiterms, referterms2, bs, param; neighbors=neighbors, m=200), varparams2)
@time gps2 = pmap(vca -> GrandPotential(:f, vca, rz, 0.0), vcas2)

@time vcas3 = pmap(param -> VCA(:A, unitcell, cluster, hilbert, origiterms, referterms3, bs, param; neighbors=neighbors, m=200), varparams3)
@time gps3 = pmap(vca -> GrandPotential(:f, vca, rz, 0.0), vcas3)


@time vcas4 = pmap(param -> VCA(:A, unitcell, cluster, hilbert, origiterms, referterms4, bs, param; neighbors=neighbors, m=200), varparams4)
@time gps4 = pmap(vca -> GrandPotential(:f, vca, rz, 0.0), vcas4)

f1 = plot(range(0, 0.2, 20), gps1 .- gps1[1], label=nothing,legend=:topright, title="Ω-Ω₀ vs M")
f2 = plot(range(0, 0.2, 20), gps2 .- gps2[1], label=nothing,legend=:topright, title="Ω-Ω₀ vs M")
f3 = plot(range(0, 0.2, 20), gps3 .- gps3[1], label=nothing,legend=:topright, title="Ω-Ω₀ vs M")
f4 = plot(range(0, 0.2, 20), gps4 .- gps4[1], label=nothing,legend=:topright, title="Ω-Ω₀ vs M")
