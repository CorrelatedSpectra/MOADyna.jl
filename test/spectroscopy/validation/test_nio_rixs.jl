# Native NiO RIXS acceptance against the tracked Quanty/PyQuanty fixtures.
# The model and onsite anchors match examples/06_nio_rixs.jl. Compare the
# raw complex response: no fitted scale, energy shift, or curve normalisation.
@testset "NiO RIXS — native vs external references" begin
    using DelimitedFiles

    ref_dir = joinpath(@__DIR__, "..", "..", "..", "docs", "dev", "validation",
                       "spectroscopy", "nio_xas", "reference")
    ref_files = joinpath.(ref_dir, ["RIXSxy_Quanty.txt", "RIXSxy.txt"])
    @test all(isfile, ref_files)
    all(isfile, ref_files) || return

    omega_in = range(-6.0, 0.0; length = 11)
    omega_loss = range(-0.5, 8.0; length = 851)
    raw_q = readdlm(ref_files[1]; skipstart = 5)
    raw_pq = readdlm(ref_files[2])
    @test size(raw_q) == size(raw_pq) == (length(omega_loss), 1 + 2length(omega_in))
    @test max(maximum(abs, raw_q[:, 1] .- omega_loss),
              maximum(abs, raw_pq[:, 1] .- omega_loss)) < 1e-12

    # One Re/Im column pair per incident energy; transpose to the public
    # scalar RIXS shape (incident energy, energy loss).
    ref_q = permutedims(complex.(raw_q[:, 2:2:end], raw_q[:, 3:2:end]))
    ref_pq = permutedims(complex.(raw_pq[:, 2:2:end], raw_pq[:, 3:2:end]))
    cut_residual(actual, reference) = maximum(
        maximum(abs, actual[k, :] .- reference[k, :]) /
        maximum(abs, reference[k, :]) for k in axes(reference, 1))

    # The references already differ by 1.53e-4 of a cut's complex peak
    # (at -3.0 eV), so a shared 1e-4 budget would reject the fixtures.
    # Keep the established 1e-4 native/Quanty acceptance budget, with 3e-4
    # for comparisons involving PyQuanty to accommodate that measured floor.
    # Each cut uses its own reference peak only as an error scale; the spectra
    # themselves retain their absolute amplitudes, including off resonance.
    @test cut_residual(ref_q, ref_pq) < 3e-4

    m = ShellModel([:Ni_2p, :Ni_3d, :L_3d])
    F2dd, F4dd = 11.14, 6.87
    F2pd, G1pd, G3pd = 6.67, 4.92, 2.80
    zeta_3d, zeta_2p = 0.081, 11.51
    tenDq, tenDqL = 0.56, 1.44
    Veg, Vt2g = 2.06, 1.21
    Udd, Upd, Delta = 7.3, 8.5, 4.7
    Bz, Hz = 1e-6, 0.120

    es_gs = onsite_energies(m;
        shells = (:Ni_3d, :L_3d), U = (Ni_3d = Udd,),
        anchors = [(Ni_3d = 8, L_3d = 10) => 0.0,
                   (Ni_3d = 9, L_3d = 9) => Delta])
    es_int = onsite_energies(m;
        U = (Ni_3d = Udd,), pairs = ((:Ni_2p, :Ni_3d) => Upd,),
        anchors = [(Ni_2p = 6, Ni_3d = 8, L_3d = 10) => 0.0,
                   (Ni_2p = 6, Ni_3d = 9, L_3d = 9) => Delta,
                   (Ni_2p = 5, Ni_3d = 9, L_3d = 10) => 0.0])

    H_common = coulomb(m, :Ni_3d; U = Udd, F = (F2dd, F4dd)) +
               zeta_3d * LS(m, :Ni_3d) +
               tenDq * Akm(m, :Ni_3d, :Oh, [0.6, -0.4]) +
               tenDqL * Akm(m, :L_3d, :Oh, [0.6, -0.4]) +
               Veg * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :Eg) +
               Vt2g * hop(m, :Ni_3d, :L_3d, :Oh; irrep = :T2g) +
               Bz * (2 * Sz(m, :Ni_3d) + Lz(m, :Ni_3d)) +
               Hz * Sz(m, :Ni_3d)
    H_final = H_common + es_gs.Ni_3d * n(m, :Ni_3d) + es_gs.L_3d * n(m, :L_3d)
    H_int = H_common +
            coulomb(m, :Ni_2p, :Ni_3d; U = Upd, F = (F2pd,), G = (G1pd, G3pd)) +
            zeta_2p * LS(m, :Ni_2p) +
            es_int.Ni_2p * n(m, :Ni_2p) + es_int.Ni_3d * n(m, :Ni_3d) +
            es_int.L_3d * n(m, :L_3d)
    T = dipole(m, :Ni_2p => :Ni_3d)

    # Include d8/d9/d10 charge-transfer configurations and both full-core
    # and single-core-hole sectors, so both absorption and emission act.
    basis_gs = basis(m, nshells(m, :Ni_2p) == 6, total(m) == 24)
    basis_rixs = basis(m, nshells(m, :Ni_2p) in 5:6, total(m) == 24)
    @test (length(basis_gs), length(basis_rixs)) == (190, 310)
    gs = eigen(H_final, basis_gs; n = 3)
    psi0 = embed(gs.vectors[:, 1], basis_gs => basis_rixs)
    H_final_sp = assemble(compile(H_final, basis_rixs), basis_rixs)
    H_int_sp = assemble(compile(H_int, basis_rixs), basis_rixs)

    # G(z) uses Gamma/2: FWHM 0.6/0.1 eV matches reference eta 0.3/0.05.
    result = rixs(H_final_sp, H_int_sp, basis_rixs, T[1], T[2], psi0;
                  omega_in_grid = omega_in, omega_out_grid = omega_loss,
                  Gamma_intermediate = 0.6, Gamma_final = 0.1,
                  Eg = gs.values[1], krylovdim = 200)
    residual_q = cut_residual(result.tensor, ref_q)
    residual_pq = cut_residual(result.tensor, ref_pq)
    @info "Native NiO RIXS maximum per-cut peak-relative residual" residual_q residual_pq
    @test residual_q < 1e-4
    @test residual_pq < 3e-4
end
