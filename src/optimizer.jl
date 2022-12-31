import VariationalMC.funcs: WaveFunction, zeros_like_params

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

function AdamOptimizer(wf::WaveFunction)
    m = zeros_like_params(wf)
    v = zeros_like_params(wf)
    t = 0
    β1 = 0.9
    β2 = 0.999
    α = 0.001
    ϵ = 1e-8
    AdamOptimizer(m, v, β1, β2, α, ϵ, t)
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
    return -1 .* adam.a .* broadcast((mh, vh) -> mh ./ (sqrt.(vh) .+ adam.epsilon), mhat, vhat)
end


mutable struct SGDOptimizer <: Optimizer
    t::Int
    SGDOptimizer() = new(0)
end

function step!(sgd::SGDOptimizer, dp_el::TupleParams)::TupleParams
    sgd.t += 1
    return -1 .* dp_el
end
