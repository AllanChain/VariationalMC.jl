using QMC
using QMC.molecule
using QMC.funcs
using LinearAlgebra
using Random
using Test

function test_grad(func, molecule::Molecule, walker::AbstractVector{Float64})
    f = log_func(func, molecule, walker)
    dx_f = dx_log(func, molecule, walker)
    dx = 1e-7
    for i = 1:size(walker, 1)
        new_walker = copy(walker)
        new_walker[i] += dx
        f2 = log_func(func, molecule, new_walker)
        @test (f2 - f) / dx ≈ dx_f[i] rtol = 1e-5
    end
end


function test_normalized_laplacian(
    func,
    molecule::Molecule,
    walker::AbstractVector{Float64},
)
    f1 = exp(log_func(func, molecule, walker))
    nl = normalized_laplacian(func, molecule, walker)
    # There will be some numerial instability if dx is too small
    dx = 1e-5
    laplacian = 0
    for i = 1:size(walker, 1)
        x = copy(walker)
        x[i] += dx
        f2 = exp(log_func(func, molecule, x))
        x[i] += dx
        f3 = exp(log_func(func, molecule, x))
        laplacian += (f1 + f3 - 2 * f2) / (f2 * dx^2)
    end
    @test laplacian ≈ nl rtol = 2e-4
end

function test_laplacian(func, molecule::Molecule, walker::AbstractVector{Float64})
    f1 = log_func(func, molecule, walker)
    ll = laplacian_log(func, molecule, walker)
    # There will be some numerial instability if dx is too small
    dx = 1e-5
    laplacian = 0
    for i = 1:size(walker, 1)
        x = copy(walker)
        x[i] += dx
        f2 = log_func(func, molecule, x)
        x[i] += dx
        f3 = log_func(func, molecule, x)
        laplacian += (f1 + f3 - 2 * f2) / (dx^2)
    end
    @test laplacian ≈ ll rtol = 2e-4
end

@testset "Slater Jastrow functions" begin
    basis = read_basis("6-31g")
    # H_basis = basis["H"]
    # molecule = Molecule([
    #     Atom(1, [0.0, 0.0, 0.0], H_basis),
    #     Atom(1, [5.0, 0.0, 0.0], H_basis),
    # ])
    # walker = randn(MersenneTwister(1), 2 * 3)
    Li_basis = basis["Li"]
    molecule = Molecule([
        Atom(3, [0.0, 0.0, 0.0], Li_basis),
        Atom(3, [5.0, 0.0, 0.0], Li_basis),
    ])
    walker = randn(MersenneTwister(1), 6 * 3)
    @testset "Slater determinant product" begin
        @testset "Derivative wrt params" begin
            slater = SlaterDetProd(molecule)
            f = log_func(slater, molecule, walker)
            dα_f, _ = dp_log(slater, molecule, walker)
            dp = 1e-7
            slater.mo_coeff_alpha[1, 2] += dp
            f2 = log_func(slater, molecule, walker)
            @test (f2 - f) / dp ≈ dα_f[1, 2] rtol = 1e-5
        end
        @testset "Derivative wrt electrons" begin
            slater = SlaterDetProd(molecule)
            test_grad(slater, molecule, walker)
        end
        @testset "Normalized laplacian" begin
            slater = SlaterDetProd(molecule)
            test_normalized_laplacian(slater, molecule, walker)
        end
        @testset "Laplacian of log" begin
            slater = SlaterDetProd(molecule)
            test_laplacian(slater, molecule, walker)
        end
    end
    @testset "Jastrow factor" begin
        @testset "Derivative wrt params" begin
            jastrow = Jastrow(1)
            f = log_func(jastrow, molecule, walker)
            db_f, = dp_log(jastrow, molecule, walker)
            db = 1e-7
            jastrow.b += db
            f2 = log_func(jastrow, molecule, walker)
            @test (f2 - f) / db ≈ db_f rtol = 1e-5

        end
        @testset "Derivative wrt electrons" begin
            jastrow = Jastrow(1)
            test_grad(jastrow, molecule, walker)
        end
        @testset "Laplacian of log" begin
            jastrow = Jastrow(1)
            test_laplacian(jastrow, molecule, walker)
        end
    end
    @testset "Slater Jastrow" begin
        @testset "Derivative wrt params" begin
            sj = SlaterJastrow(molecule)
            f = log_func(sj, molecule, walker)
            dα_f, _, _ = dp_log(sj, molecule, walker)
            dp = 1e-7
            sj.slater.mo_coeff_alpha[1, 2] += dp
            f2 = log_func(sj, molecule, walker)
            @test (f2 - f) / dp ≈ dα_f[1, 2] rtol = 1e-5
        end
        @testset "Derivative wrt electrons" begin
            sj = SlaterJastrow(molecule)
            test_grad(sj, molecule, walker)
        end
        @testset "Normalized laplacian" begin
            sj = SlaterJastrow(molecule)
            test_normalized_laplacian(sj, molecule, walker)
        end
        @testset "Laplacian of log" begin
            sj = SlaterJastrow(molecule)
            test_laplacian(sj, molecule, walker)
        end
    end
end
