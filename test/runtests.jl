using QMC
using QMC.molecule
using Test

@testset "Basis" begin
    basis = read_basis("sto-3g")
    @test length(basis["H"]) === 1
    bas = basis["H"][begin]
    @test bas.l == 0
    @test_throws "No such file or directory" read_basis("none")
end
