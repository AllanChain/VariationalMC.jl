using Configurations

export load_config, build_molecule, Config

@option struct AtomConfig
    name::String
    coord::Vector{Float64}
end

@option struct SystemConfig
    basis::String
    atoms::Vector{AtomConfig}
    spins::Vector{Int}
end

@option struct QMCConfig
    iterations::Int = 20
    batch_size::Int = 256
    ansatz::String = "slater"
    seed::Union{Int, Nothing} = nothing
end

@option struct MCMCConfig
    burn_in_steps::Int = 100
    steps::Int = 20
end

@option struct CheckpointConfig
    restore_path::String = ""
    save_path::String = ""
    save_interval::Int = 60
end

@option struct AdamConfig
    beta1::Float64 = 0.9     # Exp. decay first moment
    beta2::Float64 = 0.999   # Exp. decay second moment
    a::Float64 = 0.1         # Step size
    epsilon::Float64 = 1e-8  # Epsilon for stability
end

@option struct SGDConfig
    learning_rate::Float64 = 1.0
    decay_step::Int = 100
    decay_rate::Float64 = 0.1
end

@option struct OptimConfig
    optimizer::String = "adam"
    adam::AdamConfig = AdamConfig()
    sgd::SGDConfig = SGDConfig()
end

@option struct Config
    qmc::QMCConfig = QMCConfig()
    mcmc::MCMCConfig = MCMCConfig()
    checkpoint::CheckpointConfig = CheckpointConfig()
    optim::OptimConfig = OptimConfig()
    system::SystemConfig
end

function load_config(config_file::String; kwargs...)::Config
    if !isabspath(config_file)
        config_file = joinpath(@__DIR__, "../test/configs", config_file)
    end
    return from_toml(Config, config_file; kwargs...)
end

function build_molecule(config::Config)
    basis = read_basis(config.system.basis)
    atoms = Vector{Atom}(undef, length(config.system.atoms))
    for (i, atom_cfg) in enumerate(config.system.atoms)
        atoms[i] = Atom(atom_cfg.name, atom_cfg.coord, basis[atom_cfg.name])
    end
    spins = (config.system.spins[1], config.system.spins[2])
    return Molecule(atoms, spins)
end
