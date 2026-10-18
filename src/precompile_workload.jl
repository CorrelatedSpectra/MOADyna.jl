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
        for action in (assemble(compile(H,b),b), matrixfree(H,b;threaded=false))
            eigen(action,b;n=1,threaded=false)
            gs = eigen(action,b;n=1,dense_below=0,krylovdim=6,threaded=true)
            psi, eg = gs.vectors[:,1], gs.values[1]
            xas(action,b,probes[1],gs;
                     omega_grid=range(-0.5,1.0;length=3),Gamma=0.3,krylovdim=6,
                     reorth=:full)
            tensor = xas(action,b,probes,psi;Eg=eg,
                     omega_grid=range(-0.5,1.0;length=192),Gamma=0.3,krylovdim=6,
                     reorth=:full,threaded=true)
            rixs(action,action,b,probes[1],probes[2],psi;Eg=eg,
                          omega_in_grid=range(0.1,0.1;length=1),
                          omega_out_grid=range(0.0,0.5;length=2),
                          Gamma_intermediate=0.3,Gamma_final=0.1,krylovdim=6,
                          reorth=:full)
            scattering = rixs(action,action,b,probes,probes,psi;Eg=eg,
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
        allowed = n_fermion(h) == 2
        seed = Bases._selected_basis(b, b.states[1:b.nwords]; restrictions=allowed)
        selected_eigen(H,seed;restrictions=allowed,max_states=6,
                       residual_tol=1e-7,krylovdim=6,threaded=false)
        selected = selected_response(H,seed,probes,[1.0];restrictions=allowed,
            shifts=[0.15im],max_states=6,error_tol=1e-7,krylovdim=6,threaded=false)
        resolvent(selected,0.15im)
        to_grid(selected,[0.0];Γ=0.3)
    end
end
