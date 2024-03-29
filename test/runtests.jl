using SafeTestsets

@safetestset "Basis" begin
    include("./basis.jl")
end
@safetestset "Atomic Orbitals" begin
    include("./ao.jl")
    include("./ao_deriv.jl")
    include("./ao_laplacian.jl")
end
@safetestset "Functions" begin
    include("funcs.jl")
end
@safetestset "Config" begin
    include("config.jl")
end
