using QuantumLattices
using ExactDiagonalization
using QuantumClusterTheories
using QuadGK

function phase_by_azimuth(azs::AbstractVector, phs::AbstractVector)
    function pbz(bond::Bond)
        θ = azimuth(rcoordinate(bond))
        for i in eachindex(azs)
            any(≈(θ), azs[i]) && return phs[i]
        end
    end
    return pbz
end

function ExactNormalGreenFunction(t, μ, k, ω; η=0.05)
    ep = 2t*cos(k) + μ
    g = 1/(ω - ep + η*im)
    return [g 0 ; 0 g]
end

function ExactPxWaveGreenFunction(t, Δ, μ, k, ω; η=0.05)
    ep, pm, pp = 2t*cos(k) + μ, -2im*sin(k), 2im*sin(k)
    xi = √(ep^2 + (pm*pp)*Δ^2)
    uv, u², v², g, l = abs(Δ)/(2*xi), (1+ep/xi)/2, (1-ep/xi)/2, 1/(ω - xi + η*im), 1/(ω + xi + η*im)
    a = (g - l)
    g11, g14, g32, g33  = u²*g + v²*l, uv*pm*a, uv*pp*a, v²*g + u²*l
    g22, g23, g41, g44 = g11, g14, g32, g33
    return [g11 0 0 g14; 0 g22 g23 0; 0 g32 g33 0; g41 0 0 g44]
end

unitcell = Lattice([0, 0]; vectors=[[1, 0]])
cluster = Lattice(unitcell, (2,), ('p',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(cluster))
bs₁ = BinaryBases(1:4, 2)
bs₂ = BinaryBases(1:2,3:4, 0.0)
t = Hopping(:t, -1.0, 1) 
p = Pairing(:p, 0.3, 1, (Coupling(Index(:, FID(1, 1//2, 1)), Index(:, FID(1, -1//2, 1))) + Coupling(Index(:, FID(1, -1//2, 1)), Index(:, FID(1, 1//2, 1))))/2; amplitude = phase_by_azimuth([(0, 2π), (π,)], [1.0, -1.0]))
origiterms₁ = (t,)
referterms₁ = (t,)
origiterms₂ = (t, p)
referterms₂ = (t, p)
vca₁ = VCA(:N, unitcell, cluster, hilbert, origiterms₁, referterms₁, bs₁)
vca₂ = VCA(:A, unitcell, cluster, hilbert, origiterms₂, referterms₂, bs₂)


@testset "Perioder" begin
    @test vca₁.perioder.map₁ == [[a, a+b] for a in [1, 3], b in [1]]
    @test vca₁.perioder.map₂ == [[1, 2],]
    @test vca₁.perioder.channels == [([1, 1], [1, 1]),([1, 1], [2, 1]),([2, 1], [1, 1]),([2, 1], [2, 1])]
end

@testset "singleParticleGreenFunction" begin
    @test isapprox(singleParticleGreenFunction(:f, vca₁, [[-5π/12, 0.0],], [2.9,])[1][1], ExactNormalGreenFunction(-1.0, 0.0, -5π/12, 2.9); atol=1e-10)
    @test isapprox(singleParticleGreenFunction(:f, vca₂, [[-5π/12, 0.0],], [2.9,])[1][1], ExactPxWaveGreenFunction(-1.0, 0.3, 0.0, -5π/12, 2.9); atol=1e-10)
end

@testset "GrandPotential" begin
    rz = ReciprocalZone(reciprocals(cluster.vectors); length=10000)
    gp₁ = GrandPotential(:f, vca₁, rz, 0.0)
    gp₂ = GrandPotential(:f, vca₂, rz, 0.0)
    t = 1.0
    Δ = 0.3
    μ = 0.0
    ne = quadgk(k->-2t*cos(k) - μ , -π/2, π/2)[1]/π
    pe = quadgk(k->-2t*cos(k) - μ - √((2t*cos(k) + μ)^2 + 4*((sin(k))^2)*Δ^2), 0, π)[1]/π
    @test isapprox(gp₁, ne; atol=1e-10)
    @test isapprox(gp₂, pe; atol=1e-10)
end
