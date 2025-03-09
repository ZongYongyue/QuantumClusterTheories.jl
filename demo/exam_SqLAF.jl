"""
Square Lattice for 2×2 cluster with AF weiss field
"""
#define quantum lattice
unitcell = Lattice([0, 0]; vectors=[[1, 0],[0, 1]])
cluster = Lattice(unitcell,(2,2),('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = Sector(hilbert, SpinfulParticle(4, 0.0))
t = Hopping(:t, Complex(-1.0), 1)
U = Hubbard(:U, Complex(0.0))
origiterms = (t, U)
t_r = Hopping(:t, Complex(-1.0), 1)
af = Onsite(:af, Complex(0.0), MatrixCoupling(:, FID, :, σ"z", :); amplitude=antiferro([π, π]))
referterms = (t_r, U, af)
neighbors = Neighbors(0=>0.0, 1=>1.0)
rz = ReciprocalZone(reciprocals(cluster.vectors); length=100)
#instantiate VCA
vca = VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs; neighbors=neighbors, m=200)
#calculate grand potential
gps(x) = GrandPotential((t=-1.0,), (U=x[1], af=x[2]), :f, vca, rz, x[1]/2)
M = range(1e-9, 0.3, 25)


