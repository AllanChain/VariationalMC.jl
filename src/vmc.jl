import Base

using Statistics
using LinearAlgebra
using StructArrays
using Distances
using OrderedCollections
using Random
import Configurations: to_toml

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

    if config.qmc.seed !== nothing
        Random.seed!(config.qmc.seed)
        @info "Setting random seed to $(config.qmc.seed)."
    end

    optimizing = config.optim.optimizer != "nothing"
    ckpt_file = checkpoint.find_most_recent(
        config.checkpoint.restore_path;
        optimizing = optimizing,
    )
    if ckpt_file == ""
        if optimizing
            @warn "Checkpoint not found. Performing new VMC."
        else
            error(
                "Neither optimization nor evaluation checkpoints exist. " *
                "Cannot perform evaluation.",
            )
        end

        if config.qmc.ansatz == "slater"
            wf = SlaterDetProd(molecule)
        elseif config.qmc.ansatz == "slater-jastrow"
            wf = SlaterJastrow(molecule)
        else
            @error "Unknown wave function ansatz $(config.qmc.ansatz)"
        end

        walkers = init_walkers(config.qmc.batch_size, sum(molecule.spins))
        width::Float64 = 0.1
        width, acceptance =
            batch_mcmc_walk!(config.mcmc.burn_in_steps, wf, molecule, walkers, width)

        if config.optim.optimizer == "adam"
            optimizer = AdamOptimizer(wf, config.optim.adam)
        elseif config.optim.optimizer == "sgd"
            optimizer = SGDOptimizer(wf, config.optim.sgd)
        else
            return error("Unknown optimizer $(config.qmc.optimizer).")
        end
    else
        wf, walkers, optimizer, width = checkpoint.load(ckpt_file)
        acceptance = NaN
        @info "Checkpoint loaded from $ckpt_file"
        if !optimizing && !isa(optimizer, NothingOptimizer)
            optimizer = NothingOptimizer(wf)
            @info "Performing new evaluation."
        end
    end

    if config.checkpoint.save_path == ""
        @warn "checkpoint.save_path not provided. Saving checkpoint is skipped."
    else
        if !ispath(config.checkpoint.save_path)
            mkdir(config.checkpoint.save_path)
        end
        config_log_file = joinpath(config.checkpoint.save_path, "full-config.toml")
        if !ispath(config_log_file)
            to_toml(config_log_file, config)
        end
    end

    if optimizer.t > config.qmc.iterations
        @info "Already done $(config.qmc.iterations) iterations. Exiting."
        return
    end

    start_time = last_check_time = time()
    iterations_to_run = config.qmc.iterations - optimizer.t
    with_stats(
        config.checkpoint.restore_path,
        config.checkpoint.save_path;
        optimizing = optimizing,
    ) do stats
        for t = optimizer.t+1:config.qmc.iterations
            el, ∂p_E = local_energy_deriv_params(wf, molecule, walkers)
            ev = mean(el)
            σ²e = var(el)
            log_stats(
                stats,
                OrderedDict(
                    "t" => t,
                    "energy" => ev,
                    "var" => σ²e,
                ),
                OrderedDict(
                    "acceptance" => acceptance,
                ),
            )
            update_func!(wf, step!(optimizer, ∂p_E))
            width, acceptance =
                batch_mcmc_walk!(config.mcmc.steps, wf, molecule, walkers, width)
            if (
                config.checkpoint.save_path != "" &&
                (time() - last_check_time) > config.checkpoint.save_interval
            )
                last_check_time = time()
                ckpt_file = checkpoint.save(
                    config.checkpoint.save_path,
                    t, wf, walkers, optimizer, width;
                    optimizing = optimizing,
                )
                @info "Saved checkpoint to $ckpt_file"
            end
        end
    end
    @info "Time per iteration $((time() - start_time) / iterations_to_run)"
    if config.checkpoint.save_path != ""
        ckpt_file = checkpoint.save(
            config.checkpoint.save_path,
            optimizer.t, wf, walkers, optimizer, width;
            optimizing = optimizing,
        )
        @info "Saved checkpoint to $ckpt_file"
    end

    return wf, walkers
end

function vmc(config_file::String)
    return vmc(load_config(config_file))
end
