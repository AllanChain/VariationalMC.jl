using QMC
using QMC.molecule
using LinearAlgebra
using Test


@testset "H2 QMC" begin
    basis = read_basis("6-31g")
    H_basis = basis["H"]
    molecule = Molecule([
        Atom(1, [0.0, 0.0, 0.0], H_basis),
        Atom(1, [1.4, 0.0, 0.0], H_basis),
    ])
    walker = randn(6)
    @testset "H₂ deriv params" begin
        params = init_params(molecule)
        f = log_ψ(molecule, params, walker)
        f′ = log_ψ_deriv_params(molecule, params, walker)
        dp = 0.00001
        params.mo_coeff_alpha[1, 2] += dp
        f2 = log_ψ(molecule, params, walker)
        @test (f2 - f) / dp ≈ f′.mo_coeff_alpha[1, 2] rtol = 1e-4
    end
    @testset "H₂ kinetic energy" begin
        params = init_params(molecule)
        f1 = exp(log_ψ(molecule, params, walker))
        ke = local_kinetic_energy(molecule, params, walker)
        dx = 0.00001
        laplacian = 0
        for i = 1:6
            x = copy(walker)
            x[i] += dx
            f2 = exp(log_ψ(molecule, params, x))
            x[i] += dx
            f3 = exp(log_ψ(molecule, params, x))
            laplacian += (f1 + f3 - 2 * f2) / (f2 * dx^2)
        end
        @test -1 / 2 * laplacian ≈ ke rtol = 1e-4
    end
    @testset "H₂ potential energy" begin
        pe = local_potential_energy(molecule, walker)
        mype = 1 / 1.4
        a2 = molecule.atoms[2].coord
        x1 = walker[begin:3]
        x2 = walker[4:end]
        mype +=
            1 / norm(x1 .- x2) - 1 / norm(x1) - 1 / norm(x2) -
            1 / norm(x1 .- a2) - 1 / norm(x2 .- a2)
        @test pe == mype
    end
end
