# Importing calculation data

## Operator text files

An operator text dump can be read as a symbolic [`OperatorSum`](@ref). Supply
its Hilbert space and a mapping from the file's mode indices to the sites and
modes used in the model. The supported format has headers such as `MaxLength`
and `NFermionic modes`, followed by `Operator of Length N` blocks of ladder
chains and coefficients; see [`parse_quanty_operator`](@ref).

```julia
hilbert  = Hilbert(:s => FermionSite{10}(:s))
mode_map = i -> (:s, i + 1)           # zero-based file indices to one-based site modes
H        = read_quanty_operator("hamiltonian_dump.txt", hilbert, mode_map)
```

The imported operator supports the same algebra, basis construction, and
Hamiltonian representations as an operator built in Julia. Use
[`read_quanty_operators`](@ref) to read a directory of operator files.

## Wavefunctions and eigenvalues

[`read_quanty_wavefunction`](@ref) and [`read_quanty_wavefunctions`](@ref) read
fermionic `WaveFunction:` blocks, containing occupation bit strings and
coefficients, into vectors on a supplied basis.
[`read_quanty_eigenvalues`](@ref) reads the corresponding energies. The API
reference describes the supported file formats and mode mappings.

## Saving calculations

For subsequent calculations in MOADyna, use the HDF5 readers and writers:

- [`save_operator`](@ref) and [`load_operator`](@ref) for symbolic operators.
- [`save_eigensystem`](@ref) and [`load_eigensystem`](@ref) for eigenvalues and eigenvectors.
- [`save_response`](@ref) and [`load_response`](@ref) for response functions.
- [`save_spectra`](@ref) and [`load_spectra`](@ref) for spectroscopy tensors.
