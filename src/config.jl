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
    batch_size::Int = 32
end

@option struct MCMCConfig
    burn_in_steps::Int = 100
    steps::Int = 20
end

@option struct Config
    qmc::QMCConfig = QMCConfig()
    mcmc::MCMCConfig = MCMCConfig()
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
