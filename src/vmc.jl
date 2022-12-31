import Base

using Statistics
using LinearAlgebra
using StructArrays
using Distances

export vmc

function init_walkers(
    batch_size::Integer,
    nelectrons::Integer,
    ndim::Integer = 3,
)::Matrix{Float64}
    return ones(ndim * nelectrons, batch_size) + randn(ndim * nelectrons, batch_size)
end

function mean_∂p(∂p::StructArray)
    return Tuple(mean(c) for c in StructArrays.components(∂p))
end

function vmc(config::Config)
    molecule = build_molecule(config)
    wf = SlaterJastrow(molecule)

    if config.qmc.optimizer == "adam"
        optimizer = AdamOptimizer(wf)
    elseif config.qmc.optimizer == "sgd"
        optimizer = SGDOptimizer()
    else
        @error "Unknown optimizer $(config.qmc.optimizer). Using Adam"
        optimizer = AdamOptimizer(wf)
    end

    walkers = init_walkers(config.qmc.batch_size, sum(molecule.spins))
    width::Float64 = 0.1
    width, acceptance =
        batch_mcmc_walk!(config.mcmc.burn_in_steps, wf, molecule, walkers, width)

    for t = 1:config.qmc.iterations
        el, ∂p_E = local_energy_deriv_params(wf, molecule, walkers)
        ev = mean(el)
        σ²e = var(el)
        println("Loop $t; Energy $ev; Variance $σ²e; Acceptance $acceptance")
        update_func!(wf, step!(optimizer, ∂p_E))
        width, acceptance =
            batch_mcmc_walk!(config.mcmc.steps, wf, molecule, walkers, width)
    end

    return wf, walkers
end

function vmc(config_file::String)
    return vmc(load_config(config_file))
end
