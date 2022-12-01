using QMC
using QMC.molecule
using Test


@testset "H2 QMC" begin
    @testset "deriv params" begin
        basis = read_basis("6-31g")
        H_basis = basis["H"]
        molecule = Molecule([Atom(1, [0.0, 0.0, 0.0], H_basis), Atom(1, [1.4, 0.0, 0.0], H_basis)])
        walker = randn(6)
        params = init_params(molecule)
        f = log_ψ(molecule, params, walker)
        f′ = log_ψ_deriv_params(molecule, params, walker)
        dp = 0.00001
        params.mo_coeff_alpha[1, 2] += dp
        f2 = log_ψ(molecule, params, walker)
        @test (f2-f) / dp ≈ f′.mo_coeff_alpha[1, 2] rtol=1e-4
    end
end
