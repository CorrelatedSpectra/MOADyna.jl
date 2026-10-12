using Test, LinearAlgebra, SparseArrays, MOADyna

struct KuboThreadControlProbe{T,Ti<:Integer} <: AbstractMatrix{T}
    parent::SparseMatrixCSC{T,Ti}
    eigen_modes::Vector{Any}
end
Base.size(H::KuboThreadControlProbe) = size(H.parent)
Base.getindex(H::KuboThreadControlProbe,i::Int,j::Int) = H.parent[i,j]
function LinearAlgebra.eigen(H::KuboThreadControlProbe,b::EagerBasis;kwargs...)
    push!(H.eigen_modes,get(kwargs,:threaded,:missing))
    eigen(H.parent,b;kwargs...)
end

@testset "Kubo forwards threading to a partial eigensolve" begin
    s = FermionSite{3}(:threaded_kubo)
    b = basis(Hilbert(:threaded_kubo=>s),n_fermion([s])==1)
    H = assemble(compile(-n(s,1)+2n(s,2)+10n(s,3),b),b)
    op = cdag(s,1)*c(s,2)+cdag(s,2)*c(s,1)
    grid = range(-2.,4.;length=9)
    gamma, temperature, gap = 0.3, 0.7, 3.0
    probe = KuboThreadControlProbe(H,Any[])
    result = kubo_response(probe,b,op,op;temperature,ω_grid=grid,Γ=gamma,
                           N_eigs=2,threaded=false)
    @test probe.eigen_modes == [false]
    weights = exp.(-[0.,gap]./temperature)
    weights ./= sum(weights)
    reference = [(weights[1]-weights[2])*(inv(w+0.5im*gamma-gap)-
        inv(w+0.5im*gamma+gap)) for w in grid]
    @test result.tensor ≈ reference rtol=1e-11 atol=1e-11
    @test_throws ArgumentError kubo_response(H,b,op,op;temperature,
        ω_grid=grid,Γ=gamma,threaded=:invalid)
end
