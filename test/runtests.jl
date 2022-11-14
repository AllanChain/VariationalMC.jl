using QMC
using QMC.molecule
using Test

@testset "Basis" begin
    @testset "Read Basis STO-3G" begin
        basis = read_basis("sto-3g")
        @test length(basis["H"]) == 1
        bas = basis["H"][begin]
        @test bas.l == 0
        @test size(bas.exp) == (3,)
        @test size(bas.coeff) == (3, 1)

        @test length(basis["Li"]) == 3
        @test basis["Li"][1].l == 0
        @test basis["Li"][2].l == 0
        @test basis["Li"][3].l == 1
        for bas in basis["Li"]
            @test size(bas.exp) == (3,)
            @test size(bas.coeff) == (3, 1)
        end
    end
    @testset "Read Basis Error" begin
        @test_throws "No such file or directory" read_basis("none")
    end
    @testset "Eval H₂ Atomic Orbitals with STO-3G" begin
        basis = read_basis("sto-3g")
        H_basis = basis["H"]
        H₂ = Molecule([
            Atom(1, [0.0, 0.0, 0.0], H_basis),
            Atom(1, [1.4, 0.0, 0.0], H_basis),
        ])
        ao = eval_ao(H₂, [0.7, 0.0, 0.0])
        #=
        The results can be obtained by

        ```python
        >>> from pyscf import gto
        >>> mol = gto.M(atom="H 0 0 0; H 1.4 0 0", basis="sto-3g", unit="B")
        >>> mol.eval_gto("GTOval_sph", [[0.7, 0, 0]])
        array([[0.32583116, 0.32583116]])
        =#
        @test number_ao(H₂) == 2
        @test size(ao) == (2,)
        @test ao[1] ≈ 0.32583116 atol = 1e-7
        @test ao[2] ≈ 0.32583116 atol = 1e-7

        ao = eval_ao(H₂, [0.2, 0.3, -0.1])
        @test ao[1] ≈ 0.49840191 atol = 1e-7
        @test ao[2] ≈ 0.16824640 atol = 1e-7

        ao = eval_ao(H₂, [[0.7, 0.0, 0.0] [0.2, 0.3, -0.1]])
        @test size(ao) == (2, 2)
        @test ao[1, 1] ≈ 0.32583116 atol = 1e-7
        @test ao[1, 2] ≈ 0.49840191 atol = 1e-7
        @test ao[2, 2] ≈ 0.16824640 atol = 1e-7
    end
    @testset "Eval Li₂ Atomic Orbitals with STO-3G" begin
        basis = read_basis("sto-3g")
        Li_basis = basis["Li"]
        Li₂ = Molecule([
            Atom(3, [0.0, 0.0, 0.0], Li_basis),
            Atom(3, [5.0, 0.0, 0.0], Li_basis),
        ])
        ao = eval_ao(Li₂, [2.5, 0.0, 0.0])
        @test number_ao(Li₂) == 10
        @test size(ao) == (10,)
        @test ao[1] ≈ 0.00185821 atol = 1e-7
        @test ao[2] ≈ 0.06393277 atol = 1e-7
        @test ao[3] ≈ 0.10800787 atol = 1e-7
        @test ao[4] == 0
        @test ao[5] == 0

        ao = eval_ao(Li₂, [2.0, 1.0, -0.5])
        answer = [
            4.11361784e-03, 6.92492517e-02, 1.01581262e-01,
            5.07906310e-02, -2.53953155e-02, 7.73827318e-05,
            4.61404760e-02, -7.59554400e-02, 2.53184800e-02,
            -1.26592400e-02,
        ]
        @test all(isapprox.(ao, answer, atol = 1e-7))
    end
    @testset "Eval O Atomic Orbitals with STO-6G" begin
        basis = read_basis("sto-6g")
        O_basis = basis["O"]
        O = Molecule([
            Atom(8, [0.0, 0.0, 0.0], O_basis),
        ])
        ao = eval_ao(O, [0.1, -0.2, 0.5])
        @test number_ao(O) == 5
        @test length(ao) == 5
        answer = [0.18049887, 0.39548582, 0.12476461, -0.24952923, 0.62382307]
        @test all(isapprox.(ao, answer, atol = 1e-7))
    end
    @testset "Eval K₂ Atomic Orbitals with STO-6G" begin
        basis = read_basis("sto-6g")
        K_basis = basis["K"]
        K₂ = Molecule([
            Atom(35, [0.0, 0.0, 0.0], K_basis),
            Atom(35, [1.0, 0.0, 0.0], K_basis),
        ])
        ao = eval_ao(K₂, [0.3, 0.2, -0.1])
        @test number_ao(K₂) == 26
        @test length(ao) == 26
        answer = [
            4.37070090e-02, 1.14466826e+00, 2.05264108e-01,
            4.91655741e-03, 1.58860655e+00, 1.05907103e+00,
            -5.29535517e-01, 2.85797548e-01, 1.90531699e-01,
            -9.52658493e-02, 6.85039791e-03, 4.56693194e-03,
            -2.28346597e-03, 4.94913190e-06, 1.63818632e-01,
            2.93681270e-01, 2.20972997e-02, -2.71048007e-01,
            7.74422876e-02, -3.87211438e-02, -4.84602629e-01,
            1.38457894e-01, -6.92289470e-02, -3.63519779e-02,
            1.03862794e-02, -5.19313970e-03,
        ]
        @test all(isapprox.(ao, answer, atol = 1e-7))
    end
    @testset "Eval Br₂ Atomic Orbitals with STO-6G" begin
        basis = read_basis("sto-6g")
        Br_basis = basis["Br"]
        Br₂ = Molecule([
            Atom(35, [0.0, 0.0, 0.0], Br_basis),
            Atom(35, [1.0, 0.0, 0.0], Br_basis),
        ])
        ao = eval_ao(Br₂, [0.3, 0.2, -0.1])
        @test number_ao(Br₂) == 36
        @test length(ao) == 36
        answer = [
            5.11158411e-05, 3.98091290e-01, 1.04172136e+00,
            3.41196527e-02, 5.54897043e-01, 3.69931362e-01,
            -1.84965681e-01, 1.44971871e+00, 9.66479137e-01,
            -4.83239568e-01, 4.67417717e-02, 3.11611811e-02,
            -1.55805906e-02, 1.71050760e+00, -5.70169199e-01,
            -9.05265187e-01, -8.55253799e-01, 7.12711499e-01,
            2.26491058e-18, 3.09788226e-03, 3.70520725e-01,
            1.08572596e-01, -6.04763769e-03, 1.72789648e-03,
            -8.63948241e-04, -6.10784428e-01, 1.74509836e-01,
            -8.72549182e-02, -1.79179970e-01, 5.11942770e-02,
            -2.55971385e-02, -3.77746664e-01, -5.39638091e-02,
            -3.97239252e-01, 1.88873332e-01, 6.07092853e-01,
        ]
        @test all(isapprox.(ao, answer, atol = 1e-7)) broken = true
    end
    @testset "Laplacian of H with STO-3G" begin
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
