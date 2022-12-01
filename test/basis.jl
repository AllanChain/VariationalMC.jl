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
    @testset "Check STO-3G Basis Normalized" begin
        basis = read_basis("sto-3g", normalize = false)
        H_basis = basis["H"]
        @test check_basis_normalized(H_basis[1])
        for bas in basis["Li"]
            @test check_basis_normalized(bas)
        end
        for bas in basis["S"]
            @test check_basis_normalized(bas)
        end
    end
    @testset "cc-pVDZ basis is not normalized" begin
        basis = read_basis("cc-pvdz", normalize = false)
        H_basis = basis["H"]
        @test !check_basis_normalized(H_basis[1])
    end
    @testset "cc-pVDZ basis can be normalized" begin
        basis = read_basis("cc-pvdz", normalize = false)
        H_basis = basis["H"]
        normalize_basis!(H_basis)
        @test check_basis_normalized(H_basis[1])
    end
end
