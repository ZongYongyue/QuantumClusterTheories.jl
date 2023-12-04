"""
Square Lattice for 2×2 cluster with SC weiss field
"""
#define quantum lattice
unitcell = Lattice([0, 0]; vectors=[[1, 0], [0, 1]])
cluster = Lattice(unitcell, (2,2), ('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = BinaryBases(1:4,5:8, 0.0)
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 8.0)
μ = Onsite(:μ, -1.2)
origiterms = (t, U, μ)
neighbors = Neighbors(0=>0.0, 1=>1.0, 2=>√2)
coupling=Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1)))-Coupling(Index(:, FID(1, -1//2, 1)), Index(:, FID(1, 1//2, 1)))
s = Pairing(:s, 0.3, 0, coupling/2)
dx²y² = Pairing(:dx²y², 0.3, 1, coupling/2; amplitude=phase_by_azimuth([(0,π,2π),(π/2,3π/2)],[1.0,-1.0]))
es = Pairing(:es, 0.3, 1, coupling/2)
dxy = Pairing(:dxy, 0.3, 2, coupling/2; amplitude=phase_by_azimuth([(π/4,5π/4),(3π/4,7π/4)],[1.0, -1.0]))
refertermss = [(t, U, μ, s),(t, U, μ, dx²y²),(t, U, μ, es),(t, U, μ, dxy)]
rz = ReciprocalZone(reciprocals(cluster.vectors); length=100)
#instantiate VCA
vcas = [VCA(:A, unitcell, cluster, hilbert, origiterms, refertermss[i], bs; neighbors=neighbors, m=200) for i in 1:4]
#calculate grand potential
gps1(x) = GrandPotential((t=-1.0,), (s=x[1],), :f, vcas[1], rz, 0.0)
gps2(x) = GrandPotential((t=-1.0,), (dx²y²=x[1],), :f, vcas[2], rz, 0.0)
gps3(x) = GrandPotential((t=-1.0,), (es=x[1],), :f, vcas[3], rz, 0.0)
gps4(x) = GrandPotential((t=-1.0,), (dxy=x[1],), :f, vcas[4], rz, 0.0)
M = range(1e-9, 0.3, 25)