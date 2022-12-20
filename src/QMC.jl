module QMC

module molecule
include("basis.jl") 
include("molecule.jl")
include("gto.jl")
include("config.jl")
end

using .molecule
include("main.jl")
end
