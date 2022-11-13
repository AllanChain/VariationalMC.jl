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
    end
    @testset "Read Basis Error" begin
        @test_throws "No such file or directory" read_basis("none")
    end
    @testset "Eval Atomic Orbitals" begin
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
        @test ao[1] ≈ 0.32583116 rtol=1e-7
        @test ao[2] ≈ 0.32583116 rtol=1e-7

        ao = eval_ao(H₂, [0.2, 0.3, -0.1])
        @test ao[1] ≈ 0.49840191 rtol=1e-7
        @test ao[2] ≈ 0.16824640 rtol=1e-7
    end
end
