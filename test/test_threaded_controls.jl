using Test
using LinearAlgebra
using SparseArrays
using MOADyna

# The probe computes the real matrix product and records preparation requests.
# It detects a nested solver reverting threaded=false to its default :auto.
struct SparseThreadControlProbe{T,Ti<:Integer} <: AbstractMatrix{T}
    parent::SparseMatrixCSC{T,Ti}
    modes::Vector{Any}
end
Base.size(H::SparseThreadControlProbe) = size(H.parent)
Base.getindex(H::SparseThreadControlProbe, i::Int, j::Int) = H.parent[i,j]
Base.:*(H::SparseThreadControlProbe, X::AbstractMatrix) = H.parent*X
Base.:*(H::SparseThreadControlProbe, x::AbstractVector) = H.parent*x
function MOADyna.Bases._prepare_sparse_action(H::SparseThreadControlProbe;
                                              nrhs::Int=1, threaded=:auto)
    MOADyna.Bases._validate_sparse_threading(threaded)
    push!(H.modes,threaded)
    H
end

function sparse_control_attempt(f)
    try
        (; value=f(), error=nothing)
    catch error
        (; value=nothing, error)
    end
end

@testset "Public sparse threading controls" begin
    site = SpinSite{1}(:sparse_control_spin)
    b = basis(Hilbert(site.name=>site))
    sx, sy, sz = S(site)
    h_op = 0.8sz + 0.23sx
    A = assemble(compile(h_op,b),b)
    op = sx + 0.17sz
    O = assemble(compile(op,b),b)
    exact = eigen(Hermitian(Matrix(A)))
    state, Eg = exact.vectors[:,1], exact.values[1]
    source = O*state
    grid = range(-0.4,2.4;length=7)
    inc = range(0.3,1.8;length=2)
    loss = range(-0.3,2.3;length=5)
    ops = [op]
    tuning = (;krylovdim=5,reorth=:full,tol=1e-12)

    @testset "Eigen dense and iterative routes" begin
        values = Any[]
        for dense_below in (0,typemax(Int)), mode in (:auto,true,false)
            result = sparse_control_attempt() do
                eigen(A,b;n=1,dense_below,krylovdim=4,tol=1e-12,threaded=mode)
            end
            @test result.error === nothing
            if result.error === nothing
                E = result.value
                push!(values, (E.values[1],norm(A*E.vectors[:,1]-E.values[1]*E.vectors[:,1])))
            end
        end
        @test length(values)==6 && all(v -> abs(v[1]-Eg)<1e-11 && v[2]<1e-11,values)
        rejected = [try eigen(A,b;n=1,dense_below,threaded=:invalid); false
                    catch err; err isa ArgumentError end for dense_below in (0,typemax(Int))]
        @test all(rejected)
        symbolic = sparse_control_attempt(() -> eigen(h_op,b;n=1,threaded=false))
        @test symbolic.error === nothing
        symbolic.error === nothing && @test symbolic.value.values[1] ≈ Eg atol=1e-11
    end

    calls = [
        ("block Lanczos", (H,mode) -> begin
            L=MOADyna.Responses.block_lanczos(H,reshape(source,:,1);threaded=mode,tuning...)
            [only(L.R'*MOADyna.Responses.cf_block(L.α,L.β,w+Eg+0.15im)*L.R) for w in grid]
        end),
        ("correlator", (H,mode) -> correlator(H,b,ops,ops;state,Eg,form=:grid,
            ω=grid,Γ=0.3,threaded=mode,tuning...).data),
        ("XAS scalar source", (H,mode) -> xas(H,b,source;Eg,ω_grid=grid,Γ=0.3,
            threaded=mode,tuning...).tensor),
        ("XAS operator", (H,mode) -> xas(H,b,op,state;Eg,ω_grid=grid,Γ=0.3,
            threaded=mode,tuning...).tensor),
        ("XAS state list", (H,mode) -> xas(H,b,op,[state,state];Eg,ω_grid=grid,Γ=0.3,
            threaded=mode,tuning...).tensor),
        ("XAS Eigen", (H,mode) -> xas(H,b,op,exact;ω_grid=grid,Γ=0.3,
            threaded=mode,tuning...).tensor),
        ("XAS thermal", (H,mode) -> xas(H,b,op,exact;temperature=0.0,ω_grid=grid,
            Γ=0.3,threaded=mode,tuning...).tensor),
        ("RIXS", (H,mode) -> rixs(H,H,b,op,op,state;Eg,ω_in_grid=inc,ω_out_grid=loss,
            Γ_intermediate=0.3,Γ_final=0.2,threaded=mode,tuning...).tensor),
        ("RIXS state list", (H,mode) -> rixs(H,H,b,op,op,[state,state];Eg,
            ω_in_grid=inc,ω_out_grid=loss,Γ_intermediate=0.3,Γ_final=0.2,threaded=mode,tuning...).tensor),
        ("RIXS thermal", (H,mode) -> rixs(H,H,b,op,op,exact;temperature=0.0,
            ω_in_grid=inc,ω_out_grid=loss,Γ_intermediate=0.3,Γ_final=0.2,threaded=mode,tuning...).tensor),
        ("FY", (H,mode) -> fluorescence_yield(H,b,op,op,state;Eg,ω_in_grid=inc,
            Γ_intermediate=0.3,threaded=mode,tuning...).tensor),
        ("FY state list", (H,mode) -> fluorescence_yield(H,b,op,op,[state,state];Eg,
            ω_in_grid=inc,Γ_intermediate=0.3,threaded=mode,tuning...).tensor),
        ("FY thermal", (H,mode) -> fluorescence_yield(H,b,op,op,exact;temperature=0.0,
            ω_in_grid=inc,Γ_intermediate=0.3,threaded=mode,tuning...).tensor),
    ]
    for (label, run) in calls
        @testset "$label forwarding and values" begin
            results = [sparse_control_attempt(() -> run(A,mode)) for mode in (:auto,true,false)]
            @test all(r -> r.error===nothing,results)
            if all(r -> r.error===nothing,results)
                @test all(r -> r.value≈results[1].value,results)
            end
            probe = SparseThreadControlProbe(A,Any[])
            observed = sparse_control_attempt(() -> run(probe,false))
            @test observed.error === nothing
            @test !isempty(probe.modes) && all(mode -> mode===false,probe.modes)
            rejected = sparse_control_attempt(() -> run(A,:invalid))
            @test rejected.error isa ArgumentError
        end
    end

    @testset "Correlator automatic and charged recursive routes" begin
        paired_ops = [op,op]
        routes = [
            ("automatic ground state", mode -> correlator(A,b,ops,ops;
                form=:grid,ω=grid,Γ=0.3,threaded=mode,tuning...).data),
            ("automatic thermal ensemble", mode -> correlator(A,b,ops,ops;
                T=0.2,form=:grid,ω=grid,Γ=0.3,threaded=mode,tuning...).data),
            ("charged both channels", mode -> begin
                G=correlator(A,b,paired_ops,paired_ops;state,Eg,channel=:both,
                    addition_indices=[1],removal_indices=[2],form=:grid,
                    ω=grid,Γ=0.3,threaded=mode,tuning...)
                vcat(vec(G.addition.data),vec(G.removal.data))
            end),
        ]
        for (label,run) in routes
            @testset "$label" begin
                on, off = sparse_control_attempt(() -> run(true)), sparse_control_attempt(() -> run(false))
                @test on.error===nothing && off.error===nothing
                if on.error===nothing && off.error===nothing
                    @test on.value ≈ off.value rtol=1e-11 atol=1e-11
                end
                bad = sparse_control_attempt(() -> run(:invalid))
                @test bad.error isa ArgumentError
            end
        end
    end

    # The base scalar response has an independent full dense resolvent oracle.
    scalar = sparse_control_attempt(() -> xas(A,b,source;Eg,ω_grid=grid,Γ=0.3,
                                               threaded=true,tuning...))
    if scalar.error === nothing
        reference = [dot(source,((w+Eg+0.15im)*I-Matrix(A))\source) for w in grid]
        @test scalar.value.tensor ≈ reference rtol=1e-11 atol=1e-11
    end
end
