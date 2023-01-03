using VariationalMC

for b in (64, 128, 512)
    @info "slater-jastrow Adam batch $b"
    vmc(
        load_config(
            "N2.toml";
            checkpoint_save_path = "data/N2-sj-$b",
            checkpoint_restore_path = "data/N2-sj-$b",
            qmc_ansatz = "slater-jastrow",
            qmc_batch_size = b,
            qmc_seed = 42,
            qmc_iterations = 5000,
        ),
    )
    vmc(
        load_config(
            "N2.toml";
            checkpoint_save_path = "data/N2-sj-$b",
            checkpoint_restore_path = "data/N2-sj-$b",
            qmc_ansatz = "slater-jastrow",
            optim_optimizer = "nothing",
            qmc_batch_size = b,
            qmc_iterations = 1000,
        ),
    )
end
for a in (0.05,)
    @info "slater-jastrow Adam batch 256"
    vmc(
        load_config(
            "N2.toml";
            checkpoint_save_path = "data/N2-sj-adam-$a",
            checkpoint_restore_path = "data/N2-sj-adam-$a",
            qmc_ansatz = "slater-jastrow",
            optim_optimizer = "nothing",
            optim_adam_a = a,
            qmc_iterations = 1000,
        ),
    )
end
for b in (1024, 2048)
    @info "slater-jastrow Adam batch $b"
    vmc(
        load_config(
            "N2.toml";
            checkpoint_save_path = "data/N2-sj-$b",
            checkpoint_restore_path = "data/N2-sj-$b",
            qmc_ansatz = "slater-jastrow",
            qmc_batch_size = b,
            qmc_seed = 42,
            qmc_iterations = 3000,
        ),
    )
    vmc(
        load_config(
            "N2.toml";
            checkpoint_save_path = "data/N2-sj-$b",
            checkpoint_restore_path = "data/N2-sj-$b",
            qmc_ansatz = "slater-jastrow",
            optim_optimizer = "nothing",
            qmc_batch_size = b,
            qmc_iterations = 1000,
        ),
    )
end
