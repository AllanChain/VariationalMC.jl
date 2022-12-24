module QMC

module molecule
include("basis.jl") 
include("molecule.jl")
include("gto.jl")
end

module funcs
using QMC.molecule
include("funcs/init.jl")
include("funcs/slater.jl")
include("funcs/jastrow.jl")
include("funcs/slater-jastrow.jl")
end

using .molecule
using .funcs
include("config.jl")
include("main.jl")
end
