# Currently following Thijssen Computational Phsics, page 375

using Statistics
using Zygote

const Params = Vector{Float64}
const Electrons = Vector{Float64}
const BatchElectrons = Matrix{Float64}
const BatchLogPsi = Matrix{Float64} # Row vector


function dropdims_by(func::Function, a; dims)
    return dropdims(func(a, dims = dims), dims = dims)
end

function log_sign_psi(
    params::Params,
    electrons::BatchElectrons,
)::Tuple{BatchLogPsi,Vector{Int8}}
    alpha = params[1]
    return -alpha * electrons .^ 2, ones(Int8, size(electrons, 1))
end

function log_psi(params::Params, electrons::BatchElectrons)::BatchLogPsi
    return log_sign_psi(params, electrons)[1]
end

function local_energy(params::Params, electrons::BatchElectrons)::Vector{Float64}
    α = params[1]
    x = dropdims(electrons, dims = 1)
    return @. α + (1 / 2 - 2 * α^2) * x^2
end

function init_walkers(batch_size::Integer, ndim::Integer)
    return ones(ndim, batch_size) + randn(ndim, batch_size)
end

function mcmc_walk(
    params::Params,
    electrons::BatchElectrons,
    width::Float64,
)::Tuple{BatchElectrons,Float64}
    new_walkers = electrons + randn(size(electrons)) * width
    p = exp.(2(log_psi(params, new_walkers) - log_psi(params, electrons)))
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

    params = [0.4]
    # println(log_psi(params, Array([[0.0] [1.0] [-1.0]])))
    walkers = init_walkers(batch_size, 1)
    # println(walkers)
    el = local_energy(params, walkers)
    ev = Statistics.mean(el)
    println(ev)

    width = 0.1
    walkers, width, _ = batch_mcmc_walk(100, params, walkers, width, adjust = true)

    for i = 1:50
        el = local_energy(params, walkers)
        ev = Statistics.mean(el)
        σ²e = var(el)
        ∂p_logψ = Zygote.forwarddiff(params -> log_psi(params, walkers), params)
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
