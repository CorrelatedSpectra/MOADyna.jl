using Test
using LinearAlgebra
using SparseArrays
using Random: MersenneTwister
using MOADyna

# A zero-alpha multiplication must not inspect values in its RHS. This array
# makes that part of the mul! contract observable without a timing assertion.
struct SparseActionUnreadableRHS{T,N} <: AbstractArray{T,N}
    dims::NTuple{N,Int}
end
Base.size(X::SparseActionUnreadableRHS) = X.dims
Base.getindex(::SparseActionUnreadableRHS, I...) = error("zero-alpha RHS was read")

@testset "Prepared sparse row action" begin
    action_defined = isdefined(MOADyna.Bases, :_SparseRowAction)
    prepare_defined = isdefined(MOADyna.Bases, :_prepare_sparse_action)
    @test action_defined
    @test prepare_defined

    # Keep the initial RED run an assertion failure about the absent feature,
    # rather than an UndefVarError hiding the remaining behavioral tests.
    if action_defined && prepare_defined
        RowAction = MOADyna.Bases._SparseRowAction
        prepare = MOADyna.Bases._prepare_sparse_action
        rng = MersenneTwister(73291)

        @testset "Exact row access, precision and index types" begin
            # Samples exercise each property across precision, sparse index and
            # RHS widths; report the properties, not every repeated sample.
            checks = Dict{String,Vector{Bool}}()
            record(name, passed) = push!(get!(checks,name,Bool[]),passed)
            for T in (Float32, Float64, ComplexF32, ComplexF64), Ti in (Int32, Int64)
                rtol = T in (Float32, ComplexF32) ? 3e-6 : 2e-14
                M = randn(rng, T, 7, 7)
                M[2, :] .= 0
                M[:, 2] .= 0
                H = SparseMatrixCSC{T,Ti}(sparse(M + M'))
                H_before = copy(H)
                action = RowAction(H; nchunks=4)
                record("Hermitian row storage reuses the input", ishermitian(H) &&
                       action.parent === H && action.row_storage === H)
                record("Shape and element type are preserved", size(action)==size(H) && eltype(action)===T)
                record("Every row is owned once", collect(Iterators.flatten(action.chunks))==collect(1:size(H,1)))
                record("Preparation is idempotent", prepare(action;nrhs=3)===action)
                for width in (1,3,9)
                    X = randn(rng,T,size(H,2),width)
                    expected = Matrix(H)*X
                    actual = action*X
                    record("Block output precision", eltype(actual)===T)
                    record("Hermitian block action", isapprox(actual,expected;rtol,atol=rtol))
                    record("Empty rows produce zero", all(iszero,actual[2,:]))
                    Y = similar(expected)
                    record("In-place block action", mul!(Y,action,X)===Y && isapprox(Y,expected;rtol,atol=rtol))
                    record("Repeated action is deterministic", isequal(action*X,actual) && isequal(action*X,actual))
                end
                x = randn(rng,T,size(H,2))
                y = similar(x)
                record("Hermitian vector action", isapprox(action*x,Matrix(H)*x;rtol,atol=rtol))
                record("In-place vector action", mul!(y,action,x)===y && isapprox(y,Matrix(H)*x;rtol,atol=rtol))
                record("Hermitian input remains fixed", H==H_before)

                # A complex rectangular input detects accidental conjugation.
                G = randn(rng,T,7,5)
                G[2,:] .= 0
                G[:,4] .= 0
                A = SparseMatrixCSC{T,Ti}(sparse(G))
                A[6,4] = zero(T)
                before = copy(A)
                general = RowAction(A;nchunks=19)
                record("General row storage is a transpose copy", general.parent===A &&
                       general.row_storage!==A && Matrix(general.row_storage)==transpose(Matrix(A)))
                record("Sparse index precision", eltype(general.row_storage.rowval)===Ti)
                record("General row coverage", collect(Iterators.flatten(general.chunks))==collect(1:size(A,1)))
                for width in (1,3,9)
                    X = randn(rng,T,size(A,2),width)
                    record("General rectangular block action", isapprox(general*X,Matrix(A)*X;rtol,atol=rtol))
                end
                x = randn(rng,T,size(A,2))
                record("General rectangular vector action", isapprox(general*x,Matrix(A)*x;rtol,atol=rtol))
                record("General input entries and structure remain fixed", A==before &&
                       A.colptr==before.colptr && A.rowval==before.rowval && A.nzval==before.nzval)
            end
            for (property,samples) in sort!(collect(checks);by=first)
                @testset "$property" begin
                    @test all(samples)
                end
            end
        end

        @testset "Mixed complex RHS and noncontiguous views" begin
            H = sparse([2.0 0 -0.5; 0 1.0 0.25; -0.5 0.25 3.0])
            action = RowAction(H; nchunks=3)
            X = randn(rng, ComplexF64, 3, 3)
            @test eltype(action * X) === ComplexF64
            @test action * X ≈ Matrix(H) * X
            @test action * X[:, 1] ≈ Matrix(H) * X[:, 1]
            storage_x = randn(rng, ComplexF64, 6, 6)
            storage_y = zeros(ComplexF64, 6, 6)
            xv = view(storage_x, 1:2:6, 2)
            yv = view(storage_y, 2:2:6, 3)
            @test mul!(yv, action, xv) === yv
            @test yv ≈ Matrix(H) * xv
            Xm = view(storage_x, 1:2:6, 1:2:6)
            Ym = view(storage_y, 2:2:6, 2:2:6)
            @test mul!(Ym, action, Xm) === Ym
            @test Ym ≈ Matrix(H) * Xm
            @test all(iszero, storage_y[1:2:6, :])
        end

        @testset "Five-argument mul! coefficients and zero semantics" begin
            A = sparse(ComplexF64[1+2im 0 -3im; 2 4-im 0; 0 3+im 2])
            action = RowAction(A; nchunks=4)
            for X in (randn(rng, ComplexF64, 3), randn(rng, ComplexF64, 3, 4))
                Y = randn(rng, ComplexF64, size(X)...)
                old = copy(Y)
                alpha, beta = 0.7-0.2im, -0.4+0.1im
                @test mul!(Y, action, X, alpha, beta) === Y
                @test Y ≈ alpha * Matrix(A) * X + beta * old

                fill!(Y, complex(NaN, NaN))
                @test mul!(Y, action, X, alpha, 0) === Y
                @test Y ≈ alpha * Matrix(A) * X
                fill!(Y, complex(NaN, NaN))
                @test mul!(Y, action, X) === Y
                @test Y ≈ Matrix(A) * X

                unreadable = SparseActionUnreadableRHS{ComplexF64,ndims(X)}(size(X))
                Y .= old
                @test MOADyna.Bases._sparse_mul!(Y, action, unreadable, 0, beta) === Y
                @test Y ≈ beta * old
                fill!(Y, complex(NaN, NaN))
                @test MOADyna.Bases._sparse_mul!(Y, action, unreadable, 0, 0) === Y
                @test all(iszero, Y)
            end
        end

        @testset "Structured RHS uses unambiguous standard fallbacks" begin
            H = sparse([2.0 0.5; 0.5 3.0])
            action = RowAction(H)
            D = Diagonal([1.0, 2.0])
            U = UpperTriangular([1.0 2.0; 0.0 3.0])
            @test action * D ≈ Matrix(H) * D
            @test mul!(zeros(2, 2), action, U, 1, 0) ≈ Matrix(H) * U
            x = sparsevec([1], [2.0], 2)
            @test action * x ≈ Matrix(H) * x
        end

        @testset "Dimensions, alias rejection and empty outputs" begin
            action = RowAction(sparse([2.0 1.0; -1.0 3.0]); nchunks=4)
            @test_throws DimensionMismatch action * ones(3)
            @test_throws DimensionMismatch action * ones(3, 2)
            @test_throws DimensionMismatch mul!(zeros(3), action, ones(2))
            @test_throws DimensionMismatch mul!(zeros(2, 3), action, ones(2, 2))
            @test_throws DimensionMismatch mul!(zeros(2, 2), action, ones(3, 2), 1, 0)
            @test_throws DimensionMismatch mul!(zeros(2), action, ones(3), 0, 0)
            x = [1.0, 2.0]
            @test_throws ArgumentError mul!(x, action, x)
            X = ones(2, 3)
            @test_throws ArgumentError mul!(X, action, X, 1, 0)
            backing = ones(4)
            @test_throws ArgumentError mul!(view(backing, 2:3), action, view(backing, 1:2))
            # The solve-local plan assumes a fixed matrix. Output buffers must
            # not alias either its caller-owned values or its private row copy.
            parent = sparse([1, 2], [2, 1], [2.0, 1.0], 2, 2)
            planned = RowAction(parent; nchunks=2)
            parent_before = copy(parent)
            rows_before = copy(planned.row_storage)
            @test_throws ArgumentError mul!(nonzeros(parent), planned, ones(2))
            @test_throws ArgumentError mul!(view(nonzeros(parent), :), planned, ones(2), 1, 0)
            @test_throws ArgumentError mul!(reshape(nonzeros(parent), 2, 1), planned, ones(2, 1))
            @test_throws ArgumentError mul!(nonzeros(planned.row_storage), planned, ones(2))
            @test_throws ArgumentError mul!(reshape(nonzeros(planned.row_storage), 2, 1), planned, ones(2, 1), 1, 0)
            @test parent == parent_before
            @test planned.row_storage == rows_before
            hermitian_parent = spdiagm(0=>[2.0, 3.0])
            hermitian_plan = RowAction(hermitian_parent; nchunks=2)
            @test_throws ArgumentError mul!(nonzeros(hermitian_parent), hermitian_plan, ones(2))
            @test nonzeros(hermitian_parent) == [2.0, 3.0]
            @test_throws ArgumentError RowAction(spzeros(2, 2); nchunks=0)
            @test_throws ArgumentError RowAction(spzeros(2, 2); nchunks=-1)

            for (m, n) in ((0, 0), (0, 4), (4, 0), (4, 5))
                empty_action = RowAction(spzeros(ComplexF64, m, n); nchunks=4)
                @test empty_action * ones(ComplexF64, n) == zeros(ComplexF64, m)
                @test empty_action * ones(ComplexF64, n, 3) == zeros(ComplexF64, m, 3)
                @test size(empty_action * zeros(ComplexF64, n, 0)) == (m, 0)
            end
        end

        @testset "Preparation preserves generic and small serial inputs" begin
            dense = [1.0 2; 3 4]
            diagonal = Diagonal([1.0, 2.0])
            callable = x -> 2x
            @test prepare(dense; nrhs=3) === dense
            @test prepare(diagonal) === diagonal
            @test prepare(callable) === callable
            small = sparse(dense)
            @test prepare(small) === small
            if Threads.nthreads() == 1
                # No assumption about the automatic multi-thread threshold.
                larger = spdiagm(-1=>ones(9999), 0=>2ones(10000), 1=>ones(9999))
                @test prepare(larger; nrhs=9) === larger
            end
        end

        @testset "Explicit sparse threading controls" begin
            A = sparse([2.0 0.5; 0.5 3.0])
            ready = RowAction(A; nchunks=2)
            force = try prepare(A; threaded=true) catch err; err end
            @test force isa RowAction
            disabled = try prepare(ready; threaded=false) catch err; err end
            @test disabled === A
            automatic = try prepare(A; threaded=:auto) catch err; err end
            @test automatic === A
            retained = try prepare(ready; threaded=:auto) catch err; err end
            @test retained === ready
            forced_again = try prepare(ready; threaded=true) catch err; err end
            @test forced_again === ready
            raw_disabled = try prepare(A; threaded=false) catch err; err end
            @test raw_disabled === A
            dense = Matrix(A)
            generic_results = [try prepare(dense; threaded=mode) catch err; err end
                               for mode in (:auto,true,false)]
            @test all(result -> result === dense, generic_results)
            bad_modes = (:bogus, :on, :off, "auto", nothing, 0, 1)
            rejected = [try prepare(input; threaded=bad); false
                        catch err; err isa ArgumentError end
                        for input in (A, ready, dense), bad in bad_modes]
            @test all(rejected)
        end
    end
end
