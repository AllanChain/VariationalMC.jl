using QMC
using QMC.molecule
using QMC.funcs
using LinearAlgebra
using Random
using Test


@testset "Jastrow factor" begin
    basis = read_basis("6-31g")
    H_basis = basis["H"]
    molecule = Molecule([
        Atom(1, [0.0, 0.0, 0.0], H_basis),
        Atom(1, [1.4, 0.0, 0.0], H_basis),
    ])
    @testset "Value" begin
        walker = randn(6)
        jastrow = Jastrow(1)
        f = log_func(jastrow, molecule, walker)
        r_ij = norm(walker[1:3] - walker[4:6])
        @test f == r_ij / (1 + r_ij) / 2
    end
    @testset "Derivative wrt params" begin
        walker = randn(MersenneTwister(1), 6)
        jastrow = Jastrow(1)
        f = log_func(jastrow, molecule, walker)
        db_f, = dp_log(jastrow, molecule, walker)
        db = 1e-7
        jastrow.b += db
        f2 = log_func(jastrow, molecule, walker)
        @test (f2 - f) / db ≈ db_f rtol = 1e-5
    end
    @testset "Derivative wrt electrons" begin
        walker = randn(MersenneTwister(1), 6)
        jastrow = Jastrow(1)
        f = log_func(jastrow, molecule, walker)
        dx_f = dx_log(jastrow, molecule, walker)
        dx = 1e-7
        for i = 1:6
            new_walker = copy(walker)
            new_walker[i] += dx
            f2 = log_func(jastrow, molecule, new_walker)
            @test (f2 - f) / dx ≈ dx_f[i] rtol = 1e-5
        end
    end
    @testset "Laplacian of log" begin
        walker = randn(MersenneTwister(1), 6)
        jastrow = Jastrow(1)
        f1 = log_func(jastrow, molecule, walker)
        ll = laplacian_log(jastrow, molecule, walker)
        # There will be some numerial instability if dx is too small
        dx = 1e-5
        laplacian = 0
        for i = 1:6
            x = copy(walker)
            x[i] += dx
            f2 = log_func(jastrow, molecule, x)
            x[i] += dx
            f3 = log_func(jastrow, molecule, x)
            laplacian += (f1 + f3 - 2 * f2) / (dx^2)
        end
        @test laplacian ≈ ll rtol = 1e-4
    end
end
