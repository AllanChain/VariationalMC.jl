using VariationalMC

for mol in ("H2", "Li2", "ethanol", "Ne")
    @info "Processng $mol"
    vmc(
        load_config(
            mol * ".toml";
            checkpoint_save_path = "data/" * mol * "-sj",
            checkpoint_restore_path = "data/" * mol * "-sj",
            qmc_ansatz = "slater-jastrow",
        ),
    )
    vmc(
        load_config(
            mol * ".toml";
            checkpoint_save_path = "data/" * mol * "-sj",
            checkpoint_restore_path = "data/" * mol * "-sj",
            qmc_ansatz = "slater-jastrow",
            optim_optimizer = "nothing",
            qmc_iterations = 1000,
        ),
    )
    vmc(
        load_config(
            mol * ".toml";
            checkpoint_save_path = "data/" * mol * "-slater",
            checkpoint_restore_path = "data/" * mol * "-slater",
            qmc_ansatz = "slater",
        ),
    )
    vmc(
        load_config(
            mol * ".toml";
            checkpoint_save_path = "data/" * mol * "-slater",
            checkpoint_restore_path = "data/" * mol * "-slater",
            qmc_ansatz = "slater",
            optim_optimizer = "nothing",
            qmc_iterations = 1000,
        ),
    )
end
