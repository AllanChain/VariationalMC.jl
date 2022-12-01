using QMC
using QMC.molecule
using Test

@testset "Laplacian of Atomic Orbitals" begin
    @testset "Laplacian of H with STO-6G" begin
        basis = read_basis("sto-6g")
        H_basis = basis["H"]
        H = Molecule([Atom(1, [0.0, 0.0, 0.0], H_basis)], (1, 0))
        ao_laplacian = eval_ao_laplacian(H, [0.0, 0.0, 0.0])
        @test length(ao_laplacian) == 1
        @test ao_laplacian[1] ≈ -28.77712326
        ao_laplacian = eval_ao_laplacian(H, [0.1, 0.2, -0.1])
        @test ao_laplacian[1] ≈ -4.315583988391443
    end
    @testset "Laplacian of Li₂ with STO-6G" begin
        basis = read_basis("sto-6g")
        Li_basis = basis["Li"]
        Li₂ = Molecule([
            Atom(3, [0.0, 0.0, 0.0], Li_basis),
            Atom(3, [5.0, 0.0, 0.0], Li_basis),
        ])
        ao_laplacian = eval_ao_laplacian(Li₂, [1.0, -0.1, 0.2])
        @test length(ao_laplacian) == 10
        answer = [
            3.28607649e-01, -3.84960318e-02, -3.68131392e-01,
            3.68131392e-02, -7.36262784e-02, 3.14029995e-04,
            -9.55470781e-04, 8.80178685e-03, 2.20044671e-04,
            -4.40089343e-04,
        ]
        @test all(isapprox.(ao_laplacian, answer, atol = 1e-7))
    end
    @testset "Laplacian of K₂ with STO-6G" begin
        basis = read_basis("sto-6g")
        K_basis = basis["K"]
        K₂ = Molecule([
            Atom(35, [0.0, 0.0, 0.0], K_basis),
            Atom(35, [1.0, 0.0, 0.0], K_basis),
        ])
        ao_laplacian = eval_ao_laplacian(K₂, [0.3, 0.2, -0.1])
        @test length(ao_laplacian) == 26
        answer = [
            9.89618120e+00, -1.23777652e+01, 1.29277394e+00,
            2.70267207e-01, -3.93274666e+01, -2.62183111e+01,
            1.31091555e+01, -2.55010460e+00, -1.70006974e+00,
            8.50034868e-01, 2.84568218e-01, 1.89712145e-01,
            -9.48560726e-02, 4.76612661e-03, 2.77912232e+00,
            -1.12178939e+00, 1.91454456e-01, -3.35569802e+00,
            9.58770863e-01, -4.79385432e-01, 3.66254893e+00,
            -1.04644255e+00, 5.23221276e-01, -1.84233715e-01,
            5.26382043e-02, -2.63191022e-02,
        ]
        @test all(isapprox.(ao_laplacian, answer, atol = 1e-7))
    end
end
