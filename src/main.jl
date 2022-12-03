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

export main, QMCParams, log_ψ, log_sgn_ψ, log_ψ_deriv_params, init_params, vmc,
    local_energy, local_kinetic_energy, local_potential_energy

function Base.:-(p1::QMCParams, p2::QMCParams)
    return QMCParams(
        p1.mo_coeff_alpha - p2.mo_coeff_alpha,
        p1.mo_coeff_beta - p2.mo_coeff_beta,
    )
end
function Base.:+(p1::QMCParams, p2::QMCParams)
    return QMCParams(
        p1.mo_coeff_alpha + p2.mo_coeff_alpha,
        p1.mo_coeff_beta + p2.mo_coeff_beta,
    )
end
function Base.:*(x::T, p2::QMCParams) where {T<:Number}
    return QMCParams(x * p2.mo_coeff_alpha, x * p2.mo_coeff_beta)
end
function Base.:/(params::QMCParams, x::T) where {T<:Number}
    return QMCParams(params.mo_coeff_alpha / x, params.mo_coeff_beta / x)
end

function by_αβ(func::Function, molecule::Molecule, electrons)
    result = func(molecule, electrons)
    if ndims(result) == 2
        return result[:, begin:molecule.spins[1]], result[:, molecule.spins[1]+1:end]
    elseif ndims(result) == 3
        return result[:, :, begin:molecule.spins[1]], result[:, :, molecule.spins[1]+1:end]
    else
        throw(ErrorException("Result returns unexcepted dims $(ndims(result))"))
    end
end

function log_sgn_ψ(
    molecule::Molecule,
    params::QMCParams,
    electrons::AbstractVector{T},
)::Tuple{T,T} where {T<:Number}
    electrons = reshape(electrons, 3, :)
    ao_α, ao_β = by_αβ(eval_ao, molecule, electrons)
    log_ψα, sgn_ψα = logabsdet(params.mo_coeff_alpha * ao_α)
    log_ψβ, sgn_ψβ = logabsdet(params.mo_coeff_beta * ao_β)
    return log_ψα .+ log_ψβ, sgn_ψα .* sgn_ψβ # TODO: fix sign
end

function log_sgn_ψ(
    molecule::Molecule,
    params::QMCParams,
    batch_electrons::AbstractMatrix{T},
)::Tuple{Matrix{T},Vector{T}} where {T<:Number}
    # return broadcast(electrons -> log_sgn_ψ(params, electrons), batch_electrons)
    result = StructArray([
        log_sgn_ψ(molecule, params, electrons) for electrons in eachcol(batch_electrons)
    ])
    log_ψ, sgn_ψ = StructArrays.components(result)
    return adjoint(log_ψ), sgn_ψ
end

function log_ψ(molecule::Molecule, params::QMCParams, electrons)
    return log_sgn_ψ(molecule, params, electrons)[1]
end

function signed_minor(A)
    B = similar(A)
    m, n = size(A)
    for i = 1:m
        for j = 1:n
            B[i, j] = (-1)^(i + j) * det(A[setdiff(1:end, i), setdiff(1:end, j)])
        end
    end
    return B
end

function log_ψ_deriv_params(molecule::Molecule, params::QMCParams, electrons::Electrons)
    electrons = reshape(electrons, 3, :)
    ao_α, ao_β = by_αβ(eval_ao, molecule, electrons)
    A_α = params.mo_coeff_alpha * ao_α
    A_β = params.mo_coeff_beta * ao_β
    return QMCParams(transpose(ao_α / A_α), transpose(ao_β / A_β))
end

function log_ψ_deriv_params(
    molecule::Molecule,
    params::QMCParams,
    batch_electrons::BatchElectrons,
)
    return [
        log_ψ_deriv_params(molecule, params, electrons) for
        electrons in eachcol(batch_electrons)
    ]
end

function eval_ao_deriv_sum(molecule::Molecule, electrons)
    return dropdims(sum(eval_ao_deriv(molecule, electrons), dims = 1), dims = 1)
end

# ∇²(D↑⋅D↓)/(D↑⋅D↓) = (∇²D↑)/(D↑) + (∇²D↓)/(D↓)
function local_kinetic_energy(
    molecule::Molecule,
    params::QMCParams,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    electrons = reshape(electrons, 3, :)
    ∇²ao_α, ∇²ao_β = by_αβ(eval_ao_laplacian, molecule, electrons)
    sum_∇ao_α, sum_∇ao_β = by_αβ(eval_ao_deriv_sum, molecule, electrons)
    ao_α, ao_β = by_αβ(eval_ao, molecule, electrons)
    A_α = params.mo_coeff_alpha * ao_α
    A_β = params.mo_coeff_beta * ao_β
    inv_A_α_mo = A_α \ params.mo_coeff_alpha
    inv_A_β_mo = A_β \ params.mo_coeff_beta
    return -1 / 2 * (
        tr(inv_A_α_mo * ∇²ao_α) + tr(inv_A_α_mo * sum_∇ao_α)^2 -
        tr((inv_A_α_mo * sum_∇ao_α)^2) +
        tr(inv_A_β_mo * ∇²ao_β) + tr(inv_A_β_mo * sum_∇ao_β)^2 -
        tr((inv_A_β_mo * sum_∇ao_β)^2)
    )
end

function local_kinetic_energy(
    molecule::Molecule,
    params::QMCParams,
    batch_electrons::AbstractMatrix{T},
)::Vector{T} where {T<:Number}
    return [
        local_kinetic_energy(molecule, params, electrons) for
        electrons in eachcol(batch_electrons)
    ]
end

function local_potential_energy(
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    atoms = hcat([atom.coord for atom in molecule.atoms]...)
    charges = hcat([atom.charge for atom in molecule.atoms]...)
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

function local_energy(molecule::Molecule, params::QMCParams, electrons)
    return (
        local_kinetic_energy(molecule, params, electrons) +
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
    molecule::Molecule,
    params::QMCParams,
    electrons::BatchElectrons,
    width::Float64,
)::Tuple{BatchElectrons,Float64}
    new_walkers = electrons + randn(size(electrons)) * width
    p = exp.(2(log_ψ(molecule, params, new_walkers) - log_ψ(molecule, params, electrons)))
    cond = rand(Float64, size(p)) .< p
    # println(resize(cond, size(electrons)))
    new_walkers = ifelse.(cond, new_walkers, electrons)
    acceptance = mean(cond)
    # println("Old $electrons")
    # println("New $new_walkers")
    return new_walkers, acceptance
end


function batch_mcmc_walk(
    steps::Integer,
    molecule::Molecule,
    params::QMCParams,
    electrons::BatchElectrons,
    width::Float64;
    adjust::Bool = false,
)
    walkers = electrons
    accum_accept = Vector{Float64}(undef, steps)
    for i = 1:steps
        walkers, acceptance = mcmc_walk(molecule, params, walkers, width)
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


function init_params(molecule::Molecule)
    nao = number_ao(molecule)
    return QMCParams(
        rand(Float64, (molecule.spins[1], nao)),
        rand(Float64, (molecule.spins[2], nao)),
    )
end


function vmc(molecule::Molecule, batch_size::Integer; steps::Int = 20, burnin::Int = 20)
    params = init_params(molecule)
    walkers = init_walkers(batch_size, sum(molecule.spins))
    width = 0.1
    walkers, width, _ =
        batch_mcmc_walk(burnin, molecule, params, walkers, width, adjust = true)

    for i = 1:steps
        el = local_energy(molecule, params, walkers)
        ev = mean(el)
        σ²e = var(el)
        ∂p_logψ = log_ψ_deriv_params(molecule, params, walkers)
        ∂p_E = 2(mean(el .* ∂p_logψ) - ev * mean(∂p_logψ))
        println("Loop $i; Energy $ev; Variance $σ²e; Params $params")
        params -= ∂p_E
        walkers, width, acceptance = batch_mcmc_walk(100, molecule, params, walkers, width)
        mean_acceptance = mean(acceptance)
        if mean_acceptance > 0.55
            width *= 1.1
        elseif mean_acceptance < 0.5
            width *= 0.9
        end
    end

    return params, walkers
end

function main()
    basis = read_basis("6-31g")
    H_basis = basis["H"]
    H₂ = Molecule([Atom(1, [0.0, 0.0, 0.0], H_basis), Atom(1, [1.4, 0.0, 0.0], H_basis)])
    vmc(H₂, 256, steps = 20)
end

# main()
