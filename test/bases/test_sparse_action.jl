using Test, LinearAlgebra, SparseArrays, MOADyna

@testset "Prepared sparse row action" begin
    RowAction = MOADyna.Bases._SparseRowAction
    prepare = MOADyna.Bases._prepare_action
    H = sparse(ComplexF64[2 0 1+im; 0 0 0; 1-im 0 3])
    X = ComplexF64[1 im; 2 3; 4-im 2]
    action = RowAction(H; nchunks=4)
    @test action.row_storage === H
    @test collect(Iterators.flatten(action.chunks)) == collect(1:3)
    @test action * X ≈ Matrix(H) * X
    @test action * X[:,1] ≈ Matrix(H) * X[:,1]

    # Rectangular complex input distinguishes transpose storage from adjoint.
    A = sparse(ComplexF64[1+im 0 2; 0 0 0; 3 2-im 0; 0 4 1])
    general = RowAction(A; nchunks=3)
    @test general * X ≈ Matrix(A) * X
    alpha, beta = 0.7-0.2im, -0.4+0.1im
    Y = ones(ComplexF64,4,2)
    @test mul!(Y,general,X,alpha,beta) ≈ alpha*Matrix(A)*X .+ beta
    fill!(Y,complex(NaN,NaN))
    @test mul!(Y,general,X) ≈ Matrix(A)*X
    @test all(iszero,mul!(Y,general,fill(complex(NaN,NaN),3,2),0,0))

    # Krylov callers can pass strided views without touching adjacent storage.
    storage = zeros(ComplexF64,6,2)
    output = view(storage,2:2:6,:)
    mul!(output,action,X)
    @test output ≈ Matrix(H)*X
    @test all(iszero,storage[1:2:6,:])

    # These structured RHS previously exposed ambiguities on Julia 1.10.
    @test action * Diagonal([1.,2.,3.]) ≈ Matrix(H)*Diagonal([1.,2.,3.])
    U = UpperTriangular([1. 2 0; 0 3 1; 0 0 2])
    @test mul!(zeros(ComplexF64,3,3),action,U,1,0) ≈ Matrix(H)*U

    @test_throws DimensionMismatch mul!(zeros(4),action,ones(3))
    x = ones(3)
    @test_throws ArgumentError mul!(x,action,x)
    parent = sparse([1,2],[2,1],[2.,1.],2,2)
    guarded = RowAction(parent)
    @test_throws ArgumentError mul!(nonzeros(parent),guarded,ones(2))
    @test_throws ArgumentError mul!(nonzeros(guarded.row_storage),guarded,ones(2))
    @test RowAction(spzeros(3,0))*zeros(0) == zeros(3)

    @test prepare(H;threaded=true) isa RowAction
    @test prepare(action;threaded=false) === H
    @test prepare(H;threaded=:auto) === H
    @test prepare(action;threaded=:auto) === action
    dense = Matrix(H)
    @test prepare(dense;threaded=true) === dense
    @test_throws ArgumentError prepare(H;threaded=:invalid)
end
