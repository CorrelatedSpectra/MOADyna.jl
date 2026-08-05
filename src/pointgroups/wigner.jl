# Wigner D^ℓ from a 3×3 Cartesian rotation matrix.
#
# Active rotation (rotates physical states, §8.5):
#   D^ℓ_{m'm}(α, β, γ) = e^{-i m' α} d^ℓ_{m'm}(β) e^{-i m γ}
# Improper rotations (det R = -1) get an extra parity factor s^ℓ
# applied externally (see §7 step 5).
#
# m-ordering convention: row index m' = -ℓ..+ℓ; column m = -ℓ..+ℓ.

# Standard small-d Wigner matrix element (real for real β).
function _small_d(ℓ::Int, m′::Int, m::Int, β::Real)
    # Numerical-stability shortcut for β ≈ 0.
    if abs(β) < 1e-14
        return m′ == m ? 1.0 : 0.0
    end
    # Numerical-stability shortcut for β ≈ π:
    #   d^ℓ_{m'm}(π) = (-1)^{ℓ-m} δ_{m',-m}
    if abs(β - π) < 1e-12
        return m′ == -m ? Float64((-1)^(ℓ - m)) : 0.0
    end

    cβ2 = cos(β / 2)
    sβ2 = sin(β / 2)
    fact = ℓ_ -> Float64(factorial(big(ℓ_)))
    pre = sqrt(fact(ℓ + m′) * fact(ℓ - m′) * fact(ℓ + m) * fact(ℓ - m))

    s_lo = max(0, m - m′)
    s_hi = min(ℓ + m, ℓ - m′)
    acc = 0.0
    for s in s_lo:s_hi
        denom = fact(ℓ + m - s) * fact(s) * fact(m′ - m + s) * fact(ℓ - m′ - s)
        sign  = iseven(m′ - m + s) ? 1.0 : -1.0
        acc += sign * cβ2^(2ℓ + m - m′ - 2s) * sβ2^(m′ - m + 2s) / denom
    end
    pre * acc
end

# zyz active Euler decomposition of a proper rotation.
function _zyz_euler(R::SMatrix{3,3,Float64,9})
    # Conventions: R = R_z(α) R_y(β) R_z(γ).
    # cos β = R_zz; sin β ≥ 0 by branch choice.
    cβ = clamp(R[3, 3], -1.0, 1.0)
    β  = acos(cβ)
    if abs(sin(β)) < 1e-12
        # Gimbal: β = 0 or π. Set γ = 0, absorb into α.
        if cβ > 0
            # β = 0: R = R_z(α + γ). Pull α from upper-left rotation.
            α = atan(R[2, 1], R[1, 1])
            γ = 0.0
        else
            # β = π: R = R_z(α) diag(1,-1,-1) R_z(γ). Combined R[1,1] = cos(α-γ).
            α = atan(-R[2, 1], -R[1, 1])
            γ = 0.0
        end
    else
        α = atan(R[2, 3], R[1, 3])
        γ = atan(R[3, 2], -R[3, 1])
    end
    return α, β, γ
end

# Full D^ℓ from a Cartesian rotation matrix. Handles both proper (det=+1)
# and improper (det=-1) by factoring through the proper part and applying
# the parity factor s^ℓ (s = det R) per §7.
function _wignerd_matrix(R::SMatrix{3,3,Float64,9}, ℓ::Int)
    s = det(R)
    s_int = s > 0 ? 1 : -1
    R_proper = s_int == 1 ? R : -R
    α, β, γ = _zyz_euler(R_proper)
    n = 2ℓ + 1
    D = Matrix{ComplexF64}(undef, n, n)
    parity = s_int == 1 ? 1.0 : Float64((-1)^ℓ)
    for (i, m′) in enumerate(-ℓ:ℓ), (j, m) in enumerate(-ℓ:ℓ)
        D[i, j] = parity * cis(-m′ * α) * _small_d(ℓ, m′, m, β) * cis(-m * γ)
    end
    return D
end
