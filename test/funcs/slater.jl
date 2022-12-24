using QMC
using QMC.funcs
using QMC.molecule
using LinearAlgebra
using Random
using Test


@testset "Slater determinant product" begin
    basis = read_basis("6-31g")
    H_basis = basis["H"]
    molecule = Molecule([
        Atom(1, [0.0, 0.0, 0.0], H_basis),
        Atom(1, [1.4, 0.0, 0.0], H_basis),
    ])
    @testset "Derivative wrt params" begin
        walker = randn(MersenneTwister(1), 6)
        slater = SlaterDetProd(molecule)
        f = log_func(slater, molecule, walker)
        dα_f, _ = dp_log(slater, molecule, walker)
        dp = 1e-7
        slater.mo_coeff_alpha[1, 2] += dp
        f2 = log_func(slater, molecule, walker)
        @test (f2 - f) / dp ≈ dα_f[1, 2] rtol = 1e-5
    end
    @testset "Derivative wrt electrons" begin
        walker = randn(MersenneTwister(1), 6)
        slater = SlaterDetProd(molecule)
        f = log_func(slater, molecule, walker)
        dx_f = dx_log(slater, molecule, walker)
        dx = 1e-7
        for i = 1:6
            new_walker = copy(walker)
            new_walker[i] += dx
            f2 = log_func(slater, molecule, new_walker)
            @test (f2 - f) / dx ≈ dx_f[i] rtol = 1e-5
        end
    end
    @testset "Normalized laplacian" begin
        walker = randn(MersenneTwister(1), 6)
        slater = SlaterDetProd(molecule)
        f1 = exp(log_func(slater, molecule, walker))
        nl = normalized_laplacian(slater, molecule, walker)
        # There will be some numerial instability if dx is too small
        dx = 1e-5
        laplacian = 0
        for i = 1:6
            x = copy(walker)
            x[i] += dx
            f2 = exp(log_func(slater, molecule, x))
            x[i] += dx
            f3 = exp(log_func(slater, molecule, x))
            laplacian += (f1 + f3 - 2 * f2) / (f2 * dx^2)
        end
        @test laplacian ≈ nl rtol = 1e-4
    end
    @testset "Laplacian of log" begin
        walker = randn(MersenneTwister(1), 6)
        slater = SlaterDetProd(molecule)
        f1 = log_func(slater, molecule, walker)
        ll = laplacian_log(slater, molecule, walker)
        # There will be some numerial instability if dx is too small
        dx = 1e-5
        laplacian = 0
        for i = 1:6
            x = copy(walker)
            x[i] += dx
            f2 = log_func(slater, molecule, x)
            x[i] += dx
            f3 = log_func(slater, molecule, x)
            laplacian += (f1 + f3 - 2 * f2) / (dx^2)
        end
        @test laplacian ≈ ll rtol = 1e-4
    end
end
