using Test, LinearAlgebra, SparseArrays, MOADyna

# Detect a nested response silently reverting threaded=false to :auto.
struct SparseThreadControlProbe{T,Ti<:Integer} <: AbstractMatrix{T}
    parent::SparseMatrixCSC{T,Ti}
    modes::Vector{Any}
end
Base.size(H::SparseThreadControlProbe) = size(H.parent)
Base.getindex(H::SparseThreadControlProbe,i::Int,j::Int) = H.parent[i,j]
Base.:*(H::SparseThreadControlProbe,X::AbstractMatrix) = H.parent*X
function MOADyna.Bases._prepare_action(H::SparseThreadControlProbe;
                                              nrhs::Int=1,threaded=:auto)
    MOADyna.Bases._validate_threading(threaded)
    push!(H.modes,threaded)
    H
end

@testset "Public sparse threading controls" begin
    site = SpinSite{1}(:sparse_control_spin)
    b = basis(Hilbert(site.name=>site))
    sx, sy, sz = S(site)
    H = assemble(compile(0.8sz+0.23sx,b),b)
    op = sx+0.17sz
    O = assemble(compile(op,b),b)
    exact = eigen(Hermitian(Matrix(H)))
    state, Eg = exact.vectors[:,1], exact.values[1]
    source = O*state
    grid = range(-0.4,2.4;length=7)
    tuning = (;krylovdim=5,reorth=:full,tol=1e-12)

    E = eigen(H,b;n=1,dense_below=0,krylovdim=4,tol=1e-12,threaded=true)
    @test E.values[1] ≈ Eg atol=1e-11
    @test norm(H*E.vectors[:,1]-E.values[1]*E.vectors[:,1]) < 1e-11
    @test_throws ArgumentError eigen(H,b;n=1,threaded=:invalid)

    on = xas(H,b,source;Eg,ω_grid=grid,Γ=0.3,threaded=true,tuning...)
    off = xas(H,b,source;Eg,ω_grid=grid,Γ=0.3,threaded=false,tuning...)
    reference = [dot(source,((w+Eg+0.15im)*I-Matrix(H))\source) for w in grid]
    @test on.tensor ≈ reference rtol=1e-11 atol=1e-11
    @test off.tensor ≈ on.tensor rtol=1e-11 atol=1e-11

    # RIXS exercises both intermediate and final response preparation.
    probe = SparseThreadControlProbe(H,Any[])
    settings = (;Eg,ω_in_grid=range(0.3,1.8;length=2),ω_out_grid=grid,
                 Γ_intermediate=0.3,Γ_final=0.2,tuning...)
    serial = rixs(probe,probe,b,op,op,state;threaded=false,settings...)
    @test length(probe.modes)>=2 && all(isequal(false),probe.modes)
    parallel = rixs(H,H,b,op,op,state;threaded=true,settings...)
    @test parallel.tensor ≈ serial.tensor rtol=1e-11 atol=1e-11
end
