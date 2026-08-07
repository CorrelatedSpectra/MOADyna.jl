@testset "QuantyIO: wavefunction reader" begin

    # Helper to write a Quanty-formatted wavefunction string to a temp file.
    function _write_dump(content)
        p = tempname()
        write(p, content)
        return p
    end

    @testset "Single real wavefunction (one determinant)" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        bb = EagerBasis(h)                  # full 16-state Fock
        # Quanty mode i (0..3) → MOADyna (:s, i+1)
        mode_map = i -> (:s, i + 1)

        # Bit string "1100" = Quanty modes {0, 1} occupied.
        # In MOADyna encoding for FermionSite{4}, that's bits 0 and 1 set
        # → state index for value 0b0011 = 3.
        dump = """
        WaveFunction: Wave Function
        QComplex         =          0 (Real==0 or Complex==1)
        N                =          1 (Number of basis functions used to discribe psi)
        NFermionic modes =          4 (Number of fermions in the one particle basis)
        NBosonic modes   =          0 (Number of bosons in the one particle basis)

        #   pre-factor          Determinant
        1   1.000000000000E+00  1100
        """
        ψ = read_quanty_wavefunction(_write_dump(dump), bb, mode_map;
                                     eltype = Float64)
        @test length(ψ) == length(bb)
        # MOADyna basis state with value 0b0011 = 3 → index in sorted basis is 4
        # (since 0,1,2,3 are values 0..3 → indices 1..4 after lex sort).
        target_idx = get_index(bb, UInt64[0x0000_0000_0000_0003])
        @test target_idx > 0
        @test ψ[target_idx] ≈ 1.0
        # All other entries are zero.
        @test count(!iszero, ψ) == 1
    end

    @testset "Real superposition over two determinants" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        bb = EagerBasis(h)
        mode_map = i -> (:s, i + 1)
        dump = """
        WaveFunction: Wave Function
        QComplex         =          0 (Real==0 or Complex==1)
        N                =          2 (Number of basis functions used to discribe psi)
        NFermionic modes =          4 (Number of fermions in the one particle basis)
        NBosonic modes   =          0 (Number of bosons in the one particle basis)

        #   pre-factor          Determinant
        1   7.071067811865E-01  1100
        2   7.071067811865E-01  0011
        """
        ψ = read_quanty_wavefunction(_write_dump(dump), bb, mode_map;
                                     eltype = Float64)
        idx_1100 = get_index(bb, UInt64[0x03])  # bits 0,1 → value 3
        idx_0011 = get_index(bb, UInt64[0x0c])  # bits 2,3 → value 12
        @test ψ[idx_1100] ≈ 1 / sqrt(2)
        @test ψ[idx_0011] ≈ 1 / sqrt(2)
        @test count(!iszero, ψ) == 2
        @test ψ' * ψ ≈ 1.0 atol = 1e-12      # normalized
    end

    @testset "Complex wavefunction (QComplex=1)" begin
        s = FermionSite{4}(:s)
        h = Hilbert(:s => s)
        bb = EagerBasis(h)
        mode_map = i -> (:s, i + 1)
        dump = """
        WaveFunction: Wave Function
        QComplex         =          1 (Real==0 or Complex==1)
        N                =          3 (Number of basis functions used to discribe psi)
        NFermionic modes =          4 (Number of fermions in the one particle basis)
        NBosonic modes   =          0 (Number of bosons in the one particle basis)

        #   pre-factor           pre-factor          Determinant
        1   8.164965809277E-01   0.000000000000E+00  1100
        2   0.000000000000E+00   4.082482904639E-01  0011
        3   2.449489742783E-01  -3.265986323711E-01  1010
        """
        ψ = read_quanty_wavefunction(_write_dump(dump), bb, mode_map)
        @test eltype(ψ) <: Complex
        idx_1100 = get_index(bb, UInt64[0x03])
        idx_0011 = get_index(bb, UInt64[0x0c])
        idx_1010 = get_index(bb, UInt64[0x05])  # bits 0,2 → value 5
        @test ψ[idx_1100] ≈ ComplexF64(0.8164965809277, 0.0) atol = 1e-12
        @test ψ[idx_0011] ≈ ComplexF64(0.0, 0.4082482904639) atol = 1e-12
        @test ψ[idx_1010] ≈ ComplexF64(0.2449489742783, -0.3265986323711) atol = 1e-12
    end

    @testset "Multiple wavefunctions (Eigensystem-style dump)" begin
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        bb = EagerBasis(h, n_fermion(h) == 1)   # 2 states: |10⟩, |01⟩
        mode_map = i -> (:s, i + 1)
        dump = """
        === Eigenstate 1 (E = -1.0) ===

        WaveFunction: Wave Function
        QComplex         =          0 (Real==0 or Complex==1)
        N                =          2 (Number of basis functions used to discribe psi)
        NFermionic modes =          2 (Number of fermions in the one particle basis)
        NBosonic modes   =          0 (Number of bosons in the one particle basis)

        #   pre-factor          Determinant
        1   7.071067811865E-01  10
        2   7.071067811865E-01  01

        === Eigenstate 2 (E = 1.0) ===

        WaveFunction: Wave Function
        QComplex         =          0 (Real==0 or Complex==1)
        N                =          2 (Number of basis functions used to discribe psi)
        NFermionic modes =          2 (Number of fermions in the one particle basis)
        NBosonic modes   =          0 (Number of bosons in the one particle basis)

        #   pre-factor          Determinant
        1  -7.071067811865E-01  10
        2   7.071067811865E-01  01
        """
        ψs = read_quanty_wavefunctions(_write_dump(dump), bb, mode_map;
                                       eltype = Float64)
        @test length(ψs) == 2
        @test ψs[1]' * ψs[1] ≈ 1.0 atol = 1e-12
        @test ψs[2]' * ψs[2] ≈ 1.0 atol = 1e-12
        # The two states are orthogonal — symmetric and antisymmetric
        # combinations of the two singly-occupied determinants.
        @test ψs[1]' * ψs[2] ≈ 0.0 atol = 1e-12
    end

    @testset "Eigenvalues file" begin
        path = tempname()
        write(path, """
        # First few eigenvalues of something
        -1.234567
        0.0
        2.345678
        """)
        evs = read_quanty_eigenvalues(path)
        @test evs ≈ [-1.234567, 0.0, 2.345678]
    end

    @testset "Determinant outside basis throws" begin
        s = FermionSite{2}(:s)
        h = Hilbert(:s => s)
        bb = EagerBasis(h, n_fermion(h) == 1)   # only n=1 states; "11" not in basis
        mode_map = i -> (:s, i + 1)
        dump = """
        WaveFunction: Wave Function
        QComplex         =          0 (Real==0 or Complex==1)
        N                =          1 (Number of basis functions used to discribe psi)
        NFermionic modes =          2 (Number of fermions in the one particle basis)
        NBosonic modes   =          0 (Number of bosons in the one particle basis)

        #   pre-factor          Determinant
        1   1.000000000000E+00  11
        """
        @test_throws ArgumentError read_quanty_wavefunction(
            _write_dump(dump), bb, mode_map; eltype = Float64)
    end
end
