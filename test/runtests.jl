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
    @testset "Eval H₂ Atomic Orbitals" begin
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
        @test size(ao) == (2,)
        @test ao[1] ≈ 0.32583116 atol = 1e-7
        @test ao[2] ≈ 0.32583116 atol = 1e-7

        ao = eval_ao(H₂, [0.2, 0.3, -0.1])
        @test ao[1] ≈ 0.49840191 atol = 1e-7
        @test ao[2] ≈ 0.16824640 atol = 1e-7
    end
    @testset "Eval Li₂ Atomic Orbitals" begin
        basis = read_basis("sto-3g")
        Li_basis = basis["Li"]
        Li₂ = Molecule([
            Atom(3, [0.0, 0.0, 0.0], Li_basis),
            Atom(3, [5.0, 0.0, 0.0], Li_basis),
        ])
        ao = eval_ao(Li₂, [2.5, 0.0, 0.0])
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
            -1.26592400e-02
        ]
        @test all(isapprox.(ao, answer, atol=1e-7))
    end
end
