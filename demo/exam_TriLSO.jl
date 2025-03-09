"""
Triangular Lattice for 3 sites cluster with spiral order as weiss field
"""
#define quantum lattice
unitcell = Lattice([0, 0]; vectors=[[1, 0],[0, 1]])
cluster = Lattice([0, 0], [1/2, √3/2], [-1/2, √3/2]; vectors=[[3/2, √3/2],[0, √3]])
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = BinaryBases(6, 3)
t = Hopping(:t, Complex(-1.0), 1)
U = Hubbard(:U, Complex(0.0))
μ = Onsite(:μ, -0.0)
origiterms = (t, U, μ)
t_r = Hopping(:t, Complex(-1.0), 1)
r = Onsite(:r, Complex(0.0),coupling_by_spinrotation(π/2, π/2); amplitude=belongs([1]))
g = Onsite(:g, Complex(0.0),coupling_by_spinrotation(π/2, 7π/6); amplitude=belongs([2]))
b = Onsite(:b, Complex(0.0),coupling_by_spinrotation(π/2, 11π/6); amplitude=belongs([3]))
referterms = (t_r, U, μ, r, g, b)
neighbors = Neighbors(0=>0.0, 1=>1.0)
rz = ReciprocalZone(reciprocals(cluster.vectors); length=100)
#instantiate VCA
vca = VCA(:S, unitcell, cluster, hilbert, origiterms, referterms, bs; neighbors=neighbors, m=200)
#calculate grand potential
gps1(x) = GrandPotential((U=4.0, μ=-2.0), (U=4.0, r=x[1], g=x[1], b=x[1], μ=x[2]), :f, vca, rz, 0.0)
gps2(x) = GrandPotential((U=5.0, μ=-2.5), (U=5.0, r=x[1], g=x[1], b=x[1], μ=x[2]), :f, vca, rz, 0.0)
gps3(x) = GrandPotential((U=8.0, μ=-4.0), (U=8.0, r=x[1], g=x[1], b=x[1], μ=x[2]), :f, vca, rz, 0.0)
gps4(x) = GrandPotential((U=12.0, μ=-6.0), (U=12.0, r=x[1], g=x[1], b=x[1], μ=x[2]), :f, vca, rz, 0.0)
gps5(x) = GrandPotential((U=32.0, μ=-16.0), (U=32.0, r=x[1], g=x[1], b=x[1], μ=x[2]), :f, vca, rz, 0.0)
M1 = (1/1)*range(1e-9, 1.0, 25)
M2 = (4/5)*range(1e-9, 1.0, 25)
M3 = (1/2)*range(1e-9, 1.0, 25)
M4 = (1/3)*range(1e-9, 1.0, 25)
M5 = (1/8)*range(1e-9, 1.0, 25)

