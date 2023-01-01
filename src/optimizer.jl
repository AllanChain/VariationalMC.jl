import VariationalMC.funcs: WaveFunction, zeros_like_params
import VariationalMC.config: AdamConfig, SGDConfig

export Optimizer, AdamOptimizer, SGDOptimizer, step!

TupleParams = Tuple{Vararg{Union{AbstractArray{Float64},Float64}}}

abstract type Optimizer end

mutable struct AdamOptimizer <: Optimizer
    m::TupleParams    # First moment
    v::TupleParams    # Second moment
    beta1::Float64    # Exp. decay first moment
    beta2::Float64    # Exp. decay second moment
    a::Float64        # Step size
    epsilon::Float64  # Epsilon for stability
    t::Int            # Time step (iteration)
end

function AdamOptimizer(wf::WaveFunction, adam_config::AdamConfig)
    return AdamOptimizer(
        zeros_like_params(wf),
        zeros_like_params(wf),
        adam_config.beta1,
        adam_config.beta2,
        adam_config.a,
        adam_config.epsilon,
        0,
    )
end

function step!(adam::AdamOptimizer, dp_el::TupleParams)::TupleParams
    adam.t += 1
    adam.m =
        broadcast(x -> x .* adam.beta1, adam.m) .+
        broadcast(x -> x .* (1 - adam.beta1), dp_el)
    adam.v =
        broadcast(x -> x .* adam.beta2, adam.v) .+
        broadcast(x -> x .^ 2 .* (1 - adam.beta2), dp_el)
    mhat = adam.m ./ (1 - adam.beta1^adam.t)
    vhat = adam.v ./ (1 - adam.beta2^adam.t)
    return -1 .* adam.a .*
           broadcast((mh, vh) -> mh ./ (sqrt.(vh) .+ adam.epsilon), mhat, vhat)
end


mutable struct SGDOptimizer <: Optimizer
    t::Int
    learning_rate::Float64
    decay_step::Int
    decay_rate::Float64
end

function SGDOptimizer(::WaveFunction, sgd_config::SGDConfig)
    return SGDOptimizer(
        0,
        sgd_config.learning_rate,
        sgd_config.decay_step,
        sgd_config.decay_rate,
    )
end

function step!(sgd::SGDOptimizer, dp_el::TupleParams)::TupleParams
    sgd.t += 1
    if sgd.t % sgd.decay_step == 0
        sgd.learning_rate *= sgd.decay_rate
    end
    return -sgd.learning_rate .* dp_el
end
