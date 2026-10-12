# A tiny conserved basis exercises common concrete solver/spectroscopy types.
# Runs only while building the package cache; no process-wide settings change.
using PrecompileTools: @compile_workload

@compile_workload let
    site = FermionSite{4}(:orbital)
    h = Hilbert(:orbital => site)
    b = basis(h, n_fermion(h) == 2)
    hopping = sum(add_hc(cdag(site,i)*c(site,i+1)) for i in 1:3)
    diagonal = sum((0.31i + 0.03i^2)*n(site,i) for i in 1:4)
    probes = [n(site,i) for i in 1:2]
    for complex_case in (false, true)
        H = diagonal - 0.4hopping
        if complex_case
            H += 0.07im*(cdag(site,1)*c(site,3)-cdag(site,3)*c(site,1))
        end
        matrix = assemble(compile(H,b),b)
        eigen(matrix,b;n=1,threaded=false)
        gs = eigen(matrix,b;n=1,dense_below=0,krylovdim=6,threaded=true)
        psi, eg = gs.vectors[:,1], gs.values[1]
        xas(matrix,b,probes[1],gs;
                     omega_grid=range(-0.5,1.0;length=3),Gamma=0.3,krylovdim=6,
                     reorth=:full)
        tensor = xas(matrix,b,probes,psi;Eg=eg,
                     omega_grid=range(-0.5,1.0;length=3),Gamma=0.3,krylovdim=6,
                     reorth=:full,threaded=true)
        rixs(matrix,matrix,b,probes[1],probes[2],psi;Eg=eg,
                          omega_in_grid=range(0.1,0.1;length=1),
                          omega_out_grid=range(0.0,0.5;length=2),
                          Gamma_intermediate=0.3,Gamma_final=0.1,krylovdim=6,
                          reorth=:full)
        scattering = rixs(matrix,matrix,b,probes,probes,psi;Eg=eg,
                          omega_in_grid=range(0.1,0.1;length=1),
                          omega_out_grid=range(0.0,0.5;length=2),
                          Gamma_intermediate=0.3,Gamma_final=0.1,krylovdim=6,
                          reorth=:full)
        polarization = ComplexF64[1,im]/sqrt(2)
        polarise(tensor,polarization)
        polarise(scattering,polarization,conj.(polarization))
        re_broaden(tensor;Gamma=0.4)
        re_broaden(scattering;Gamma_final=0.2)

    end
end
