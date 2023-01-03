using VariationalMC

molecule = build_molecule(load_config("N2.toml"))
wf = VariationalMC.funcs.SlaterJastrow(molecule)
walkers = VariationalMC.init_walkers(32, 14)

VariationalMC.batch_mcmc_walk!(1, wf, molecule, walkers, 0.1)
@profview VariationalMC.batch_mcmc_walk!(10, wf, molecule, walkers, 0.1)
