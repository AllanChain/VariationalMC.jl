module VariationalMC

module molecule
include("basis.jl")
include("molecule.jl")
include("gto.jl")
end

module funcs
using ..molecule
include("funcs/init.jl")
include("funcs/slater.jl")
include("funcs/jastrow.jl")
include("funcs/slater-jastrow.jl")
end

module config
using ..molecule
include("config.jl")
end

module optimizer
include("optimizer.jl")
end

module checkpoint
include("checkpoint.jl")
end

using .molecule
using .funcs
using .config
using .optimizer
import .checkpoint
export load_config

include("stats.jl")
include("hamiltonian.jl")
include("mcmc.jl")
include("vmc.jl")
end
