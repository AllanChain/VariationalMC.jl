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
    # Adam params
    β1 = 0.9
    β2 = 0.999
    α = 0.001
    ϵ = 1e-8

    molecule = build_molecule(config)
    wf = SlaterJastrow(molecule)

    m::Tuple{Matrix{Float64},Matrix{Float64},Float64} = (
        zeros(size(wf.slater.mo_coeff_alpha)),
        zeros(size(wf.slater.mo_coeff_beta)),
        0,
    )

    v = deepcopy(m)
    walkers = init_walkers(config.qmc.batch_size, sum(molecule.spins))
    width::Float64 = 0.1
    width, acceptance =
        batch_mcmc_walk!(config.mcmc.burn_in_steps, wf, molecule, walkers, width)

    for t = 1:config.qmc.iterations
        el, ∂p_E = local_energy_deriv_params(wf, molecule, walkers)
        ev = mean(el)
        σ²e = var(el)
        println("Loop $t; Energy $ev; Variance $σ²e; Acceptance $acceptance")
        m = broadcast(x -> x .* β1, m) .+ broadcast(x -> x .* (1 - β1), ∂p_E)
        v = broadcast(x -> x .* β2, v) .+ broadcast(x -> x .^ 2 .* (1 - β2), ∂p_E)
        mhat = m ./ (1 - β1^t)
        vhat = v ./ (1 - β2^t)
        dp = α .* broadcast((mh, vh) -> mh ./ (sqrt.(vh) .+ ϵ), mhat, vhat)
        update_func!(wf, -1 .* dp)
        width, acceptance =
            batch_mcmc_walk!(config.mcmc.steps, wf, molecule, walkers, width)
    end

    return wf, walkers
end

function vmc(config_file::String)
    return vmc(load_config(config_file))
end
