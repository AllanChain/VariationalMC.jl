# Currently following Thijssen Computational Phsics, page 375

using Statistics
using LinearAlgebra
using StructArrays
using Zygote

const Params = Vector{Float64}
const Electrons = AbstractArray{Float64,1}
const BatchElectrons = Matrix{Float64}
const BatchLogPsi = Matrix{Float64} # Row vector


function dropdims_by(func::Function, a; dims)
    return dropdims(func(a, dims = dims), dims = dims)
end

function log_sgn_ψ(params::Params, electrons::Electrons)::Tuple{Float64,Int8}
    α = params[1]
    # x = reshape(electrons, 3, :)
    # r1 = norm(x[1])
    # r2 = norm(x[2])
    # r12 = norm(x[1] - x[2])
    # return -2r1 - 2r2 + r12 / 2 / (1 + α * r12), Int8(1)
    return - α * norm(electrons), Int8(1)
end

function batch_log_sgn_ψ(
    params::Params,
    batch_electrons::BatchElectrons,
)::Tuple{BatchLogPsi,Vector{Int8}}
    # return broadcast(electrons -> log_sgn_ψ(params, electrons), batch_electrons)
    result = StructArray([
        log_sgn_ψ(params, electrons) for electrons in eachcol(batch_electrons)
    ])
    log_ψ, sgn_ψ = StructArrays.components(result)
    return adjoint(log_ψ), sgn_ψ
end

batch_log_ψ = (params, batch_electrons) -> batch_log_sgn_ψ(params, batch_electrons)[1]

function local_energy(params::Params, electrons::Electrons)::Float64
    α = params[1]
    # x = reshape(electrons, 3, :)
    # r1 = norm(x[1])
    # r2 = norm(x[2])
    # r12 = norm(x[1] - x[2])
    # return -4 + 1 / r12 - 1 / 4 / (1 + α * r12)^4 - 1 / r12 / (1 + α * r12)^3 +
    #        dot(x[1] / r1 - x[2] / r2, x[1] - x[2]) / r12 / (1 + α * r12)^2
    r = norm(electrons)
    return - 1/ r - α * (α-2/r)/2
end

function batch_local_energy(
    params::Params,
    batch_electrons::BatchElectrons,
)::Vector{Float64}
    # return broadcast(electrons -> local_energy(params, electrons), batch_electrons)
    return [local_energy(params, electrons) for electrons in eachcol(batch_electrons)]
end

function init_walkers(batch_size::Integer, nelectrons::Integer, ndim::Integer = 3)
    return ones(ndim * nelectrons, batch_size) + randn(ndim * nelectrons, batch_size)
end

function mcmc_walk(
    params::Params,
    electrons::BatchElectrons,
    width::Float64,
)::Tuple{BatchElectrons,Float64}
    new_walkers = electrons + randn(size(electrons)) * width
    p = exp.(2(batch_log_ψ(params, new_walkers) - batch_log_ψ(params, electrons)))
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
    params::Params,
    electrons::BatchElectrons,
    width::Float64;
    adjust::Bool = false,
)
    walkers = electrons
    accum_accept = Vector{Float64}(undef, steps)
    for i = 1:steps
        walkers, acceptance = mcmc_walk(params, walkers, width)
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



function main(batch_size::Integer)

    params = [0.1]
    # println(log_psi(params, Array([[0.0] [1.0] [-1.0]])))
    walkers = init_walkers(batch_size, 1)
    el = batch_local_energy(params, walkers)
    ev = Statistics.mean(el)

    width = 0.1
    walkers, width, _ = batch_mcmc_walk(5000, params, walkers, width, adjust = true)

    for i = 1:50
        el = batch_local_energy(params, walkers)
        ev = Statistics.mean(el)
        σ²e = var(el)
        ∂p_logψ = Zygote.forwarddiff(params -> batch_log_ψ(params, walkers), params)
        el = reshape(el, (1, batch_size))
        ∂p_E =
            2(
                dropdims_by(mean, el .* ∂p_logψ, dims = 2) -
                ev .* dropdims_by(mean, ∂p_logψ, dims = 2)
            )
        println("Loop $i; Energy $ev; Variance $σ²e; Params $params")
        params = params .- ∂p_E
        walkers, width, acceptance = batch_mcmc_walk(100, params, walkers, width)
        mean_acceptance = mean(acceptance)
        if mean_acceptance > 0.55
            width *= 1.1
        elseif mean_acceptance < 0.5
            width *= 0.9
        end
    end
end
# 𝜕αlogψ = Zygote.gradient(() -> log_psi(params, walkers), Params([params]))
# println(𝜕αlogψ[params])
# println(Zygote.forwarddiff(params -> log_psi(params, walkers), params))
# println(mcmc_walk(params, walkers, 0.1))

main(1024)
