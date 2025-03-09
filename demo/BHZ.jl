using QuantumLattices
using Plots
colorbar = cgrad(:linear_tritanopic_krjcw_5_98_c46_n256, rev = true);
using ExactDiagonalization
using QuantumClusterTheories

using LinearAlgebra: det, tr

function detG(G)
	dg=zeros(length(G),length(G[1]))
	for (i,gs) in enumerate(G)
	
		for (j,g) in enumerate(gs)
			dg[i,j]=abs(det(g))
		end
	end
	return log10.(dg)
end


unitcell = Lattice([0,0]; vectors=[[1.0,0], [0,1.0]])
cluster = Lattice([0,0], [0, 1],[1,1],[1,0]; vectors=[[2.0,0.0], [0.0,2.0]])
#unitcell =cluster

#table = Table(Hilbert(site=>Fock{:f}(2, 2) for site=1:2))
hilbert = Hilbert(site=>Fock{:f}(2, 2) for site=1:length(cluster))

bs = BinaryBases(16,8)#Sector(hilbert, SpinfulParticle(6, 0.0))

function pip_potential(bond::Bond)
    ϕ = azimuth(rcoordinate(bond))
	(isapprox(ϕ,0.0) || isapprox(ϕ,2*pi)) && return [Coupling(1.0im,:, FID, (1,2), :, :),Coupling(1.0im,:, FID, (2,1), :, :)]
	isapprox(ϕ,pi) && return [Coupling(-1.0im,:, FID, (1,2), :, :),Coupling(-1.0im,:, FID, (2,1), :, :)]
	isapprox(ϕ,pi/2) && return [Coupling(Complex(1.0),:, FID, (1,2), :, :),Coupling(Complex(-1.0),:, FID, (2,1), :, :)]
	isapprox(ϕ,3*pi/2) && return [Coupling(Complex(-1.0),:, FID, (1,2), :, :),Coupling(Complex(1.0),:, FID, (2,1), :, :)]
	error()
end

#return Complex(1.0)*MatrixCoupling(:, FID, σ"x",:, :)


#azimuth(rcoordinate(bonds(cluster,1)[7]))
#isapprox
#bonds(cluster,2)[7]
#pip_potential(bonds(cluster,2)[13])

t = Hopping(:t, 1.0, 1,MatrixCoupling(:, FID, σ"z",:, :))
M = Onsite(:M, 1.0,MatrixCoupling(:, FID, σ"z",:, :) )

tsoc = Hopping(:s, 0.0+0.0im, 1,pip_potential )

#V=Coulomb(:V, 1.2,1)
Uv=0.0
#uaub=1.0
U = Hubbard(:U, Uv )
U1=InterOrbitalInterSpin(:V1,Uv)
U2=InterOrbitalIntraSpin(:V2,Uv)

#u = Onsite(:u, -Uv/2,amplitude=bond->isodd(bond.points[1].site) ? 1.0 : uaub )

origiterms = (t,M)
referterms = (t,M)


#k_path = ReciprocalPath(reciprocals(unitcell), line"Γ₁-Γ₂", length=100)
#k_path = ReciprocalPath(reciprocals(unitcell), (0//1,0//1)=>(0//1,1//2),(0//1,1//2)=>(1//3,2//3),(1//3,2//3)=>(0//1,0//1), length=100)
k_path = ReciprocalPath(reciprocals(unitcell), (0//1,0//1)=>(0//1,1//1),(0//1,1//1)=>(1//1,0//1), length=150)


vca=VCA(:N, unitcell, cluster, hilbert, origiterms, referterms, bs)

ω_range = range(-6, 6, length=300)
G = singleParticleGreenFunction(:f, vca, k_path, ω_range;μ=U.value/2)

A = spectrum(G;select=1:2)
fu = plot(k_path, ω_range, A; xlabel="k", ylabel="ω", color=colorbar, title="Spectral Function",clims=(0, 10))
A = spectrum(G;select=3:4)
fd = plot(k_path, ω_range, A; xlabel="k", ylabel="ω", color=colorbar, title="Spectral Function",clims=(0, 10))


#dg=detG(G)
function specre(gfpathv::AbstractVector; select::AbstractVector=Vector(1:size(gfpathv[1][1],2)))
	A = zeros(Float64, length(gfpathv), length(gfpathv[1]))
		for i in eachindex(gfpathv)
			for j in eachindex(gfpathv[i])
				A[i, j] = (tr(gfpathv[i][j][select, select])).re
			end
		end
		return A
end
dg = specre(G; select=3:4)
colorbarg = cgrad([:yellow, :red, :LightCoral, :white, :DeepSkyBlue, :blue, :MidnightBlue],[0.1,0.25,0.45,0.55,0.75,0.9],rev = true)
f1 = plot(k_path, ω_range, dg; xlabel="k", ylabel="ω", color=colorbarg, title="Spectral Function",clims=(-10, 10))
savefig(fu,"G12.pdf")
savefig(fd,"G34.pdf")
#=

function spinspectrum(G)
	up=zeros(length(G),length(G[1]))
	down=zeros(length(G),length(G[1]))
	for (i,gs) in enumerate(G)
	
		for (j,g) in enumerate(gs)
			up[i,j]= g
		end
	end
	return log10.(dg)
end
=#