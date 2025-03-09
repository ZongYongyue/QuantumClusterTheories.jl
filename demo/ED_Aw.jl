using QuantumLattices
using ExactDiagonalization
using LinearAlgebra

unitcell = Lattice([0, 0]; vectors=[[1, 0], [0, 1]])
cluster = Lattice(unitcell, (2,2), ('p','p'))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = Sector(hilbert, SpinfulParticle(4, 0.0))
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 4.0)
μ = Onsite(:μ, -0.0)
origiterms = (t, U, μ)
referterms = (t, U, μ)
neighbors = Neighbors(0=>0.0, 1=>1.0)

table = Table(hilbert, Metric(EDKind(hilbert), hilbert))
origibonds = bonds(cluster, 1)
referbonds = filter(bond -> isintracell(bond), origibonds)

parts = Partition(:N, table, bs)
origigenerator, refergenerator = OperatorGenerator(origiterms, origibonds, hilbert; table = table), OperatorGenerator(referterms, referbonds, hilbert; table = table)
eds = EDSolver(EDKind(hilbert), parts, refergenerator, bs, table)

wr = range(-10,10,300)
cgfs = [-imag(tr(ClusterGreenFunction(true, :f, eds, w+0.05*im))) for w in wr]