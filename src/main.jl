import Base

using Statistics
using LinearAlgebra
using StructArrays
using Distances

export vmc, local_energy, local_kinetic_energy, local_potential_energy

function local_kinetic_energy(
    wf::WaveFunction,
    molecule::Molecule,
    electrons,
)
    return -1 / 2 * normalized_laplacian(wf, molecule, electrons)
end

function local_potential_energy(
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    atoms = hcat([atom.coord for atom in molecule.atoms]...)
    charges = [atom.charge for atom in molecule.atoms]
    electrons = reshape(electrons, 3, :)
    r_ae = pairwise(Euclidean(), atoms, electrons, dims = 2)
    r_ee = pairwise(Euclidean(), electrons, dims = 2)
    potential_energy = -sum(charges ./ r_ae) + sum(triu(1 ./ r_ee, 1))
    if length(molecule.atoms) > 1
        r_aa = pairwise(Euclidean(), atoms, dims = 2)
        potential_energy += sum(triu(charges .* reshape(charges, 1, :) / r_aa, 1))
    end
    return potential_energy
end

function local_potential_energy(
    molecule::Molecule,
    batch_electrons::AbstractMatrix{T},
)::Vector{T} where {T<:Number}
    return [
        local_potential_energy(molecule, electrons) for
        electrons in eachcol(batch_electrons)
    ]
end

function local_energy(wf::WaveFunction, molecule::Molecule, electrons)
    el = (
        local_kinetic_energy(wf, molecule, electrons) +
        local_potential_energy(molecule, electrons)
    )
    q1, q3 = quantile(el, [0.25, 0.75])
    iqr = q3 - q1
    clamp!(el, q1 - 3 * iqr, q3 + 3 * iqr)
    return el
end

function local_energy_deriv_params(wf::WaveFunction, molecule::Molecule, electrons)
    el = local_energy(wf, molecule, electrons)
    ev = mean(el)
    ∂p_logψ = dp_log(wf, molecule, electrons)
    mean_∂p_logψ = mean_∂p(∂p_logψ)
    StructArrays.foreachfield(v -> v .*= el, ∂p_logψ)
    mean_el_∂p_logψ = mean_∂p(∂p_logψ)
    return el, 2 .* (mean_el_∂p_logψ .- ev .* mean_∂p_logψ)
end

function init_walkers(
    batch_size::Integer,
    nelectrons::Integer,
    ndim::Integer = 3,
)::Matrix{Float64}
    return ones(ndim * nelectrons, batch_size) + randn(ndim * nelectrons, batch_size)
end

function mcmc_walk(
    wf::WaveFunction,
    molecule::Molecule,
    electrons::AbstractMatrix{Float64},
    width::Float64,
)::Tuple{AbstractMatrix{Float64},Float64}
    new_walkers = electrons + randn(size(electrons)) * width
    p = exp.(2(log_func(wf, molecule, new_walkers) - log_func(wf, molecule, electrons)))
    cond = rand(Float64, size(p)) .< p
    new_walkers = ifelse.(cond, new_walkers, electrons)
    acceptance = mean(cond)
    return new_walkers, acceptance
end


function batch_mcmc_walk!(
    steps::Integer,
    wf::WaveFunction,
    molecule::Molecule,
    electrons::AbstractMatrix{Float64},
    width::Float64;
)::Tuple{Float64,Float64}
    nthreads = Threads.nthreads()
    nwalkers = size(electrons, 2)
    perchunk = ceil(Int64, nwalkers / nthreads)
    accum_accept = Matrix{Float64}(undef, nthreads, steps)

    Threads.@threads for n = 1:nthreads
        if n == nthreads
            idx_walkers = (n-1)*perchunk+1:nwalkers
        else
            idx_walkers = (n-1)*perchunk+1:n*perchunk
        end
        walkers = electrons[:, idx_walkers]
        accum_accept_n = Vector{Float64}(undef, steps)
        for i = 1:steps
            walkers, accum_accept_n[i] =
                mcmc_walk(wf, molecule, walkers, width)
        end
        electrons[:, idx_walkers] = walkers
        accum_accept[n, :] = accum_accept_n
    end
    acceptance = mean(accum_accept)
    if acceptance > 0.55
        return width * 1.1, acceptance
    elseif acceptance < 0.5
        return width * 0.9, acceptance
    end
    return width, acceptance
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
