# =====================================================================
# Pre-dispatch input validation for `eigen(H, basis; ...)`
# =====================================================================
#
# Catches dimension / range /
# eltype mismatches up front so the user sees an actionable message
# rather than a downstream LAPACK info code or KrylovKit dimension panic.

@inline function _validate_eigen_inputs(H::AbstractMatrix, basis::EagerBasis;
                                        n::Int, which::Symbol,
                                        krylovdim::Int,
                                        x0::Union{Nothing, AbstractVector})
    N = length(basis)

    size(H, 1) == size(H, 2) ||
        throw(DimensionMismatch(
            "H is $(size(H, 1))×$(size(H, 2)); must be square."))

    size(H, 1) == N ||
        throw(DimensionMismatch(
            "H size $(size(H, 1)) does not match length(basis) = $N. " *
            "Did you assemble H against a different basis?"))

    1 ≤ n ≤ N ||
        throw(ArgumentError("n=$n outside 1:$N (length(basis))"))

    which ∈ (:SR, :LR, :LM) ||
        throw(ArgumentError(
            "which=$(repr(which)) not supported. Hermitian eigenvalues " *
            "are real, so :SI/:LI are meaningless; :SM (interior " *
            "eigenvalues) requires shift-invert and is not implemented. " *
            "Use :SR (smallest), :LR (largest), or :LM (largest magnitude)."))

    krylovdim ≥ n + 2 ||
        throw(ArgumentError(
            "krylovdim=$krylovdim < n+2 = $(n+2); too tight for Krylov."))

    if x0 !== nothing
        length(x0) == N ||
            throw(DimensionMismatch(
                "x0 has length $(length(x0)); expected $N."))

        # Eltype rules:
        #   - Only `Real` or `Complex` element types are accepted (otherwise
        #     KrylovKit will fail downstream with an opaque MethodError;
        #     better to reject up front).
        #   - Complex x0 + real H  → reject (would force the Krylov subspace
        #                                    complex and violate the contract
        #                                    that eigvecs match eltype(H)).
        #   - Anything else        → promoted to Vector{eltype(H)} at the call
        #                            site in _eigen_krylov.
        eltype(x0) <: Number ||
            throw(ArgumentError(
                "x0 has element type $(eltype(x0)); only Real or Complex " *
                "subtypes of Number are supported."))
        if eltype(x0) <: Complex && !(eltype(H) <: Complex)
            throw(ArgumentError(
                "x0 is Complex but H is real ($(eltype(H))). MOADyna's " *
                "contract is that eigenvectors match eltype(H); a " *
                "complex x0 would force the Krylov subspace complex. " *
                "Pass `real.(x0)` or use a complex H if you genuinely " *
                "need complex eigenvectors."))
        end
    end
    return nothing
end
