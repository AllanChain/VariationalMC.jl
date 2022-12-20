using QMC
using QMC.molecule
using Test

@testset "Atomic Orbitals" begin
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
    @testset "Eval Sc atomic orbitals with STO-3G" begin
        basis = read_basis("sto-3g")
        Sc = Molecule([Atom(21, [0.0, 0.0, 0.0], basis["Sc"])], (1, 0))
        ao = eval_ao(Sc, [0.1, 0.2, -0.1])
        answer = [
            3.48405195e-01, 2.09124605e+00, 2.00827116e-01,
            6.12552449e-05, 1.44335049e+00, 2.88670098e+00,
            -1.44335049e+00, 1.54038700e-01, 3.08077400e-01,
            -1.54038700e-01, 8.15032643e-03, 1.63006529e-02,
            -8.15032643e-03, 6.05324023e-03, -6.05324023e-03,
            -2.62112991e-03, -3.02662011e-03, -4.53993017e-03,
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
        println(isapprox.(ao, answer, atol = 1e-7))
        @test all(isapprox.(ao, answer, atol = 1e-7)) broken = true
    end
    @testset "Eval atomic orbitals of H with 6-31G" begin
        basis = read_basis("6-31g")
        H_basis = basis["H"]
        H = Molecule([Atom(1, [0.0, 0.0, 0.0], H_basis)], (1, 0))
        ao = eval_ao(H, [0.1, 0.2, -0.1])
        answer = [0.77699339, 0.17963399]
        @test all(isapprox.(ao, answer, atol = 1e-7))
    end
    @testset "Eval atomic orbitals of Li with 6-31G" begin
        basis = read_basis("6-31g")
        Li_basis = basis["Li"]
        Li = Molecule([Atom(3, [0.0, 0.0, 0.0], Li_basis)], (1, 0))
        ao = eval_ao(Li, [0.1, 0.2, -0.1])
        answer = [
            1.28320882, -0.01938129, 0.05872944, 0.0197182, 0.0394364,
            -0.0197182, 0.00222745, 0.0044549, -0.00222745,
        ]
        @test all(isapprox.(ao, answer, atol = 1e-7))
    end
    @testset "Eval atomic orbitals of H with cc-pVDZ" begin
        basis = read_basis("cc-pvdz")
        H_basis = basis["H"]
        H = Molecule([Atom(1, [0.0, 0.0, 0.0], H_basis)], (1, 0))
        ao = eval_ao(H, [0.1, 0.2, -0.1])
        answer = [0.62897326, 0.14604979, 0.09160394, 0.18320789, -0.09160394]
        @test all(isapprox.(ao, answer, atol = 1e-7))
    end
    @testset "Eval atomic orbitals of Li with cc-pVDZ" begin
        basis = read_basis("cc-pvdz")
        Li_basis = basis["Li"]
        Li = Molecule([Atom(3, [0.0, 0.0, 0.0], Li_basis)], (1, 0))
        ao = eval_ao(Li, [0.1, 0.2, -0.1])
        answer = [
            1.27919477e+00, -4.06104439e-01, 4.87673126e-02,
            1.94528168e-02, 3.89056336e-02, -1.94528168e-02,
            1.34665434e-03, 2.69330868e-03, -1.34665434e-03,
            1.46435482e-03, -1.46435482e-03, -6.34084236e-04,
            -7.32177409e-04, -1.09826611e-03,
        ]
        println(isapprox.(ao, answer, atol = 1e-7))
        println(ao - answer)
        @test all(isapprox.(ao, answer, atol = 1e-7)) broken = true
    end
end
