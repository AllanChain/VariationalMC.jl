using QMC
using QMC.molecule
using Test

@testset "Config Loading" begin
    @testset "Read H₂" begin
        config = load_config(joinpath(@__DIR__, "configs/H2.toml"))
        @test config.mcmc.steps == 100
        @test config.system.basis == "6-31g"
        @test length(config.system.atoms) == 2
    end
    @testset "Build H₂" begin
        config = load_config(joinpath(@__DIR__, "configs/H2.toml"))
        H₂ = build_molecule(config)
        @test H₂.spins == (1, 1)
        @test length(H₂.atoms) == 2
        @test H₂.atoms[2].coord == [1.4, 0, 0]
    end
end
