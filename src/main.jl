import Base

using Statistics
using LinearAlgebra
using StructArrays
using Distances

const Electrons = AbstractVector{Float64}
const BatchElectrons = AbstractMatrix{Float64}
mutable struct QMCParams
    mo_coeff_alpha::AbstractMatrix{Float64}
    mo_coeff_beta::AbstractMatrix{Float64}
end

export QMCParams, vmc, local_energy, local_kinetic_energy, local_potential_energy

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
    return (
        local_kinetic_energy(wf, molecule, electrons) +
        local_potential_energy(molecule, electrons)
    )
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
    electrons::BatchElectrons,
    width::Float64,
)::Tuple{BatchElectrons,Float64}
    new_walkers = electrons + randn(size(electrons)) * width
    p = exp.(2(log_func(wf, molecule, new_walkers) - log_func(wf, molecule, electrons)))
    cond = rand(Float64, size(p)) .< p
    new_walkers = ifelse.(cond, new_walkers, electrons)
    acceptance = mean(cond)
    return new_walkers, acceptance
end


function batch_mcmc_walk(
    steps::Integer,
    wf::WaveFunction,
    molecule::Molecule,
    electrons::BatchElectrons,
    width::Float64;
    adjust::Bool = false,
)
    walkers = electrons
    accum_accept = Vector{Float64}(undef, steps)
    for i = 1:steps
        walkers, acceptance = mcmc_walk(wf, molecule, walkers, width)
        accum_accept[i] = acceptance
        if adjust
            if acceptance > 0.55
                width *= 1.1
            elseif acceptance < 0.5
                width *= 0.9
            end
        end
    end
    return walkers, width, accum_accept
end

function mean_∂p(∂p::StructArray)
    return Tuple(mean(c) for c in StructArrays.components(∂p))
end

function vmc(config::Config)
    molecule = build_molecule(config)
    wf = SlaterJastrow(molecule)
    walkers = init_walkers(config.qmc.batch_size, sum(molecule.spins))
    width = 0.1
    walkers, width, _ =
        batch_mcmc_walk(
            config.mcmc.burn_in_steps,
            wf,
            molecule,
            walkers,
            width,
            adjust = true,
        )

    for i = 1:config.qmc.iterations
        el = local_energy(wf, molecule, walkers)
        ev = mean(el)
        σ²e = var(el)
        println("Loop $i; Energy $ev; Variance $σ²e")
        ∂p_logψ = dp_log(wf, molecule, walkers)
        mean_∂p_logψ = mean_∂p(∂p_logψ)
        StructArrays.foreachfield(v -> v .*= el, ∂p_logψ)
        mean_el_∂p_logψ = mean_∂p(∂p_logψ)
        ∂p_E = 2 .* (mean_el_∂p_logψ .- ev .* mean_∂p_logψ)
        update_func!(wf, -1 .* ∂p_E)
        walkers, width, acceptance =
            batch_mcmc_walk(config.mcmc.steps, wf, molecule, walkers, width)
        mean_acceptance = mean(acceptance)
        if mean_acceptance > 0.55
            width *= 1.1
        elseif mean_acceptance < 0.5
            width *= 0.9
        end
    end

    return wf, walkers
end

function vmc(config_file::String)
    return vmc(load_config(config_file))
end
