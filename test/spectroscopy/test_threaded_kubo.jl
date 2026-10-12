# Sparse action controls across direct Kubo and correlator-based observables.
using Test, LinearAlgebra, SparseArrays, Logging, MOADyna

struct KuboThreadControlProbe{T,Ti<:Integer} <: AbstractMatrix{T}
    parent::SparseMatrixCSC{T,Ti}
    prepare_modes::Vector{Any}
    eigen_modes::Vector{Any}
end
Base.size(H::KuboThreadControlProbe) = size(H.parent)
Base.getindex(H::KuboThreadControlProbe,i::Int,j::Int) = H.parent[i,j]
Base.:*(H::KuboThreadControlProbe,X::AbstractMatrix) = H.parent*X
Base.:*(H::KuboThreadControlProbe,x::AbstractVector) = H.parent*x
function MOADyna.Bases._prepare_sparse_action(H::KuboThreadControlProbe;
                                              nrhs::Int=1,threaded=:auto)
    MOADyna.Bases._validate_sparse_threading(threaded)
    push!(H.prepare_modes,threaded)
    H
end
function LinearAlgebra.eigen(H::KuboThreadControlProbe,b::EagerBasis;kwargs...)
    push!(H.eigen_modes,get(kwargs,:threaded,:missing))
    eigen(H.parent,b;kwargs...)
end

function kubo_control_attempt(f)
    try
        value = with_logger(NullLogger()) do
            f()
        end
        (;value,error=nothing)
    catch error
        (;value=nothing,error)
    end
end

@testset "Threading across remaining spectroscopy routes" begin
    s=FermionSite{3}(:threaded_kubo)
    b=basis(Hilbert(:threaded_kubo=>s),n_fermion([s])==1)
    energies=[-1.0,2.0,10.0]
    hop=energies[1]*n(s,1)+energies[2]*n(s,2)+energies[3]*n(s,3)
    H=assemble(compile(hop,b),b)
    op=cdag(s,1)*c(s,2)+cdag(s,2)*c(s,1)
    grid=range(-2.0,4.0;length=9)
    gamma,temperature=0.3,0.7
    gap=energies[2]-energies[1]
    response(weight_difference)=[weight_difference*(inv(w+0.5im*gamma-gap)-
        inv(w+0.5im*gamma+gap)) for w in grid]

    @testset "Kubo full dense mode validates and preserves the pair sum" begin
        weights=exp.(-(energies.-minimum(energies))./temperature)
        weights./=sum(weights)
        reference=response(weights[1]-weights[2])
        results=[kubo_control_attempt(() -> kubo_response(H,b,op,op;
            temperature,ω_grid=grid,Γ=gamma,threaded=mode)) for mode in (:auto,true,false)]
        @test all(r -> r.error===nothing,results)
        if all(r -> r.error===nothing,results)
            @test all(r -> isapprox(r.value.tensor,reference;rtol=1e-11,atol=1e-11),results)
        end
        invalid=kubo_control_attempt(() -> kubo_response(H,b,op,op;
            temperature,ω_grid=grid,Γ=gamma,threaded=:invalid))
        @test invalid.error isa ArgumentError
    end

    @testset "Kubo projected response forwards to its eigensolve" begin
        weights=exp.(-(energies[1:2].-energies[1])./temperature)
        weights./=sum(weights)
        reference=response(weights[1]-weights[2])
        for mode in (true,false)
            probe=KuboThreadControlProbe(H,Any[],Any[])
            result=kubo_control_attempt(() -> kubo_response(probe,b,op,op;
                temperature,ω_grid=grid,Γ=gamma,N_eigs=2,threaded=mode))
            @test result.error===nothing
            @test probe.eigen_modes==[mode]
            if result.error===nothing
                @test result.value.tensor ≈ reference rtol=1e-11 atol=1e-11
            end
        end
    end

    @testset "Partial Kubo spectrum reaches the iterative ED branch" begin
        # Above the actual dense threshold, but only three distinct eigenvalues:
        # the targeted partial solve is cheap and needs no dense pair matrices.
        bigsite=FermionSite{11}(:threaded_kubo_iterative)
        bigbasis=basis(Hilbert(:threaded_kubo_iterative=>bigsite))
        diagonal=vcat([-1.0,2.0],fill(10.0,length(bigbasis)-2))
        large=spdiagm(0=>diagonal)
        for mode in (true,false)
            probe=KuboThreadControlProbe(large,Any[],Any[])
            result=kubo_control_attempt(() -> MOADyna.Spectroscopy._kubo_spectrum(
                probe,bigbasis,2,1e-8;threaded=mode))
            @test result.error===nothing
            @test probe.eigen_modes==[mode]
            if result.error===nothing
                values,vectors=result.value
                @test values ≈ [-1.0,2.0] atol=1e-10
                @test norm(large*vectors-vectors*Diagonal(values))<1e-9
            end
        end
    end

    @testset "Conductivity and structure factor preserve explicit policy" begin
        exact=eigen(Hermitian(Matrix(H)))
        state,Eg=exact.vectors[:,1],exact.values[1]
        tuning=(;krylovdim=5,reorth=:full)
        for observable in (optical_conductivity,dynamical_structure_factor)
            on=kubo_control_attempt(() -> observable(H,b,op;ω_grid=grid,Γ=gamma,
                ψ₀=state,Eg,threaded=true,tuning...))
            probe=KuboThreadControlProbe(H,Any[],Any[])
            off=kubo_control_attempt(() -> observable(probe,b,op;ω_grid=grid,Γ=gamma,
                ψ₀=state,Eg,threaded=false,tuning...))
            @test on.error===nothing && off.error===nothing
            @test !isempty(probe.prepare_modes) && all(mode -> mode===false,probe.prepare_modes)
            if on.error===nothing && off.error===nothing
                @test on.value.tensor ≈ off.value.tensor rtol=1e-11 atol=1e-11
            end
            invalid=kubo_control_attempt(() -> observable(H,b,op;ω_grid=grid,Γ=gamma,
                threaded=:invalid,tuning...))
            @test invalid.error isa ArgumentError
            thermal=[kubo_control_attempt(() -> observable(H,b,op;ω_grid=grid,Γ=gamma,
                temperature,N_states=2,threaded=mode,tuning...)) for mode in (true,false)]
            @test all(r -> r.error===nothing,thermal)
            if all(r -> r.error===nothing,thermal)
                @test thermal[1].value.tensor ≈ thermal[2].value.tensor rtol=1e-11 atol=1e-11
            end
        end
    end
end
