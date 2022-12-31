module VariationalMC

module molecule
include("basis.jl") 
include("molecule.jl")
include("gto.jl")
end

module funcs
using VariationalMC.molecule
include("funcs/init.jl")
include("funcs/slater.jl")
include("funcs/jastrow.jl")
include("funcs/slater-jastrow.jl")
end

using .molecule
using .funcs
include("config.jl")
include("hamiltonian.jl")
include("mcmc.jl")
include("vmc.jl")
end
