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
    iterations::Int
    batch_size::Int
end

@option struct MCMCConfig
    burn_in_steps::Int
    steps::Int
end

@option struct Config
    qmc::QMCConfig = QMCConfig(; iterations = 50, batch_size = 32)
    mcmc::MCMCConfig = MCMCConfig(; burn_in_steps = 100, steps = 100)
    system::SystemConfig
end

function load_config(config_file::String)::Config
    return from_toml(Config, config_file)
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
