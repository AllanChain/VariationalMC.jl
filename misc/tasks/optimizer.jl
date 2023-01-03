using VariationalMC

# Quick check of optimizers
for a in (0.1, 0.05)
    @info "slater-jastrow adam $a"
    vmc(
        load_config(
            "N2.toml";
            checkpoint_save_path = "data/N2-sj-adam-$a",
            checkpoint_restore_path = "data/N2-sj-adam-$a",
            qmc_ansatz = "slater-jastrow",
            optim_optimizer = "adam",
            optim_adam_a = a,
            qmc_seed = 42,
            qmc_iterations = 5000,
        ),
    )
end
for a in (0.01, 0.001)
    @info "slater-jastrow adam $a"
    vmc(
        load_config(
            "N2.toml";
            checkpoint_save_path = "data/N2-sj-adam-$a",
            checkpoint_restore_path = "data/N2-sj-adam-$a",
            qmc_ansatz = "slater-jastrow",
            optim_optimizer = "adam",
            optim_adam_a = a,
            qmc_seed = 42,
            qmc_iterations = 8000,
        ),
    )
end
for r in (0.1, 0.8, 0.9, 1)
    @info "slater-jastrow SGD $r"
    vmc(
        load_config(
            "N2.toml";
            checkpoint_save_path = "data/N2-sj-sgd-$r",
            checkpoint_restore_path = "data/N2-sj-sgd-$r",
            qmc_ansatz = "slater-jastrow",
            optim_optimizer = "sgd",
            optim_sgd_decay_rate = r,
            qmc_seed = 42,
            qmc_iterations = 5000,
        ),
    )
end
