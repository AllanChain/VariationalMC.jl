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

module optimizer
include("optimizer.jl")
end

module checkpoint
include("checkpoint.jl")
end

using .molecule
using .funcs
using .optimizer
import .checkpoint
include("config.jl")
include("stats.jl")
include("hamiltonian.jl")
include("mcmc.jl")
include("vmc.jl")
end
