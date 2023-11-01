using QuantumLattices
using ExactDiagonalization
using QuantumClusterTheories
using Plots
using Distributed
colorbar = cgrad(:linear_tritanopic_krjcw_5_98_c46_n256, rev = true)

unitcell = Lattice([0, 0]; vectors=[[1, 0]])
cluster = Lattice(unitcell, (4,), ('p',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = BinaryBases([1:4,5:8], 0.0)
#bs = BinaryBases(8)
t = Hopping(:t, -1.0, 1)
coupling=Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1)))+Coupling(Index(:, FID(1, -1//2, 1)), Index(:, FID(1, 1//2, 1)))#Coupling(Index(:, FID(1, 0//1, 1)), Index(:, FID(1, 0//1, 1)))
function px(bond::Bond)
    θ = azimuth(rcoordinate(bond))
    any(≈(θ),(0, 2π)) && return 1.0
    any(≈(θ),(π,)) && return -1.0
end
p = Pairing(:p, 0.2, 1, coupling/2; amplitude=px)
μ = Onsite(:μ, 0.5)
origiterms = (t, μ, p)
referterms = (t, μ, p)
neighbors = Neighbors(0=>0.0, 1=>1.0)
@time vca = VCA(:A, unitcell, cluster, hilbert, origiterms, referterms, bs)
k_path = ReciprocalPath(reciprocals(unitcell.vectors), line"X₂-Γ-X", length=300)
ω_range = range(-3, 3, length=400)
fq = ω_range .+ (0.05*im)
@time G = singleParticleGreenFunction(:f, vca, k_path, fq)
A = spectrum(G)
c1 = plot(k_path, ω_range, A; xlabel="k", ylabel="ω", color=colorbar , title="Spectral Function",clims=(0, 10))

#=
cell = Lattice([0, 0]; vectors=[[1, 0]])
unitcell = Lattice(cell, (4,), ('p',))
cluster = Lattice(cell, (4,), ('p',))
unitcell = Lattice([0, 0]; vectors=[[1, 0]])
cluster = Lattice(unitcell, (4,), ('p',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = BinaryBases(1:4, 2)⊗BinaryBases(5:8, 2)
t = Hopping(:t, -1.0, 1)
μ = Onsite(:μ, 2.0)
origiterms = (t, μ)
referterms = (t, μ)
neighbors = Neighbors(0=>0.0, 1=>1.0)
@time vca = VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs)
k_path = ReciprocalPath(reciprocals(unitcell.vectors), line"X₂-Γ-X", length=300)
ω_range = range(-3, 3, length=400)
fq = ω_range .+ (0.05*im)
@time G = singleParticleGreenFunction(:f, vca, k_path, fq)
A = spectrum(G)
f2 = plot(k_path, ω_range, A; xlabel="k", ylabel="ω", color=colorbar , title="Spectral Function",clims=(0, 3))
=#
#=
unitcell = Lattice([0, 0]; vectors=[[1, 0]])
cluster = Lattice(unitcell, (4,))
cell = Lattice([0, 0]; vectors=[[1, 0]])
unitcell = Lattice(cell, (4,), ('p',))
cluster = Lattice(cell, (4,), ('p',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs = BinaryBases([1:4,5:8], 0.0)
#bs = BinaryBases(8)
t = Hopping(:t, -1.0, 1)
coupling=Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1)))+Coupling(Index(:, FID(1, -1//2, 1)), Index(:, FID(1, 1//2, 1)))
function px(bond::Bond)
    θ = azimuth(rcoordinate(bond))
    any(≈(θ),(0, 2π)) && return 1.0
    any(≈(θ),(π,)) && return -1.0
end
p = Pairing(:p, 1.0, 1, coupling; amplitude = px)
μ = Onsite(:μ, 0.0)
referterms = (t, μ, p)
neighbors = Neighbors(0=>0.0, 1=>1.0)

table = Table(hilbert, Metric(EDKind(typeof(referterms)), hilbert))
origibonds = bonds(cluster, neighbors)
referbonds = origibonds#filter(bond -> isintracell(bond), origibonds)
refergenerator = OperatorGenerator(referterms, referbonds, hilbert; table = table)

solver = EDSolver(EDKind(typeof(referterms)), :A, refergenerator, bs, table; m = 200)

ω_range = range(-3, 3, length=600)
fq = ω_range .+ (0.05*im)
cgfv = [ClusterGorkovGreenFunction(:f, solver, ω) for ω in fq]
A = clusterspectrum(cgfv)
a = plot(ω_range, A)
=#

#=
t = -1.0
Δ = 0.2
μ = -0.1
L = 400
p = range(-π, π, L)
y1 = sqrt.((2*t*cos.(p) .+μ).^2+4*((abs(Δ))^2)*((sin.(p)).^2))
y2 = -sqrt.((2*t*cos.(p) .+μ).^2+4*((abs(Δ))^2)*((sin.(p)).^2))

m1 = [reverse(y1[1:(L÷4)])..., reverse(y1[(3*(L÷4)+1):(L)])...]
m2 = y1[((L÷4)+1):3*(L÷4)]
m3 = y2[((L÷4)+1):3*(L÷4)]
m4 = [reverse(y2[1:(L÷4)])..., reverse(y2[(3*(L÷4)+1):(L)])...]

t1 = [reverse(m1[1:(L÷8)])..., reverse(m1[(3*(L÷8)+1):(L÷2)])...]
t2 = m1[((L÷8)+1):3*(L÷8)]
t3 = [reverse(m2[1:(L÷8)])..., reverse(m2[(3*(L÷8)+1):(L÷2)])...]
t4 = m2[((L÷8)+1):3*(L÷8)]
t5 = [reverse(m3[1:(L÷8)])..., reverse(m3[(3*(L÷8)+1):(L÷2)])...]
t6 = m3[((L÷8)+1):3*(L÷8)]
t7 = [reverse(m4[1:(L÷8)])..., reverse(m4[(3*(L÷8)+1):(L÷2)])...]
t8 = m4[((L÷8)+1):3*(L÷8)]

q = range(-π,π,(L÷4))
y = plot(q, t1)
plot!(q, t2)
plot!(q, t3)
plot!(q, t4)
plot!(q, t5)
plot!(q, t6)
plot!(q, t7)
plot!(q, t8)
=#