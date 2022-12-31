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

