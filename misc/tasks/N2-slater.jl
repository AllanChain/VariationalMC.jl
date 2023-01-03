using VariationalMC

b = 2048
@info "slater Adam batch $b"
vmc(
    load_config(
        "N2.toml";
        checkpoint_save_path = "data/N2-slater-$b",
        checkpoint_restore_path = "data/N2-slater-$b",
        qmc_ansatz = "slater",
        qmc_batch_size = b,
        qmc_seed = 42,
        qmc_iterations = 3000,
    ),
)
vmc(
    load_config(
        "N2.toml";
        checkpoint_save_path = "data/N2-slater-$b",
        checkpoint_restore_path = "data/N2-slater-$b",
        qmc_ansatz = "slater",
        optim_optimizer = "nothing",
        qmc_batch_size = b,
        qmc_iterations = 1000,
    ),
)
