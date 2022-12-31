### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 44220f0a-73c3-11ed-2cd4-4d9cd3f98b6a
begin
	using Plots
	using Printf
	using Statistics
	using Measures
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ dafcbf36-a777-46b6-a710-838167e452c6
begin
	using VariationalMC
	using VariationalMC.molecule
end

# ╔═╡ e62bd613-cde1-43e3-848c-872e7d1a3f5a
begin
	config = load_config("H2.toml"; qmc_iterations=1)
	mol = build_molecule(config)
	wf, walkers = vmc(config)
end

# ╔═╡ 186d7956-4ea8-44b0-b753-971d2d40c36e
begin
	electrons = reshape(walkers, 3, :, config.qmc.batch_size)
	walkers_ee = dropdims(sum((electrons[:, 1, :] - electrons[:, 2, :]).^2, dims=1), dims=1)
	walkers_r1 = dropdims(sum(electrons.^2, dims=1), dims=1)
	walkers_r2 = dropdims(sum((electrons .- [1.4, 0, 0]).^2, dims=1), dims=1)
	walkers_r1_min = dropdims(minimum(walkers_r1, dims=1), dims=1)
	walkers_r2_min = dropdims(minimum(walkers_r2, dims=1), dims=1)
	walkers_r_min = min.(walkers_r1_min, walkers_r2_min)
end

# ╔═╡ 8c3a1b22-c2d2-437a-86b8-0abe8d5fcee4
begin
	gr()
	
	el = local_energy(wf, mol, walkers)
	eea = plot(walkers_r_min, el, seriestype=:scatter, label="")
	plot!(xscale=:log10, minorgrid=true)
	xlabel!("Closest elec-atom distance")
	ylabel!("Energy")
	ev = mean(el)
	hline!([ev], label="")

	eee = plot(walkers_ee, el, seriestype=:scatter, label="")
	plot!(xscale=:log10, minorgrid=true)
	xlabel!("Closest elec-elec distance")
	ylabel!("Energy")
	hline!([ev], label="")

	kel = local_kinetic_energy(wf, mol, walkers)
	pel = local_potential_energy(mol, walkers)
	
	keea = plot(walkers_r_min, kel, seriestype=:scatter, label="")
	plot!(xscale=:log10, minorgrid=true)
	xlabel!("Closest elec-atom distance")
	ylabel!("Kinetic energy")

	peea = plot(walkers_r_min, pel, seriestype=:scatter, label="")
	plot!(xscale=:log10, minorgrid=true)
	xlabel!("Closest elec-atom distance")
	ylabel!("Potential energy")

	keee = plot(walkers_ee, kel, seriestype=:scatter, label="")
	plot!(xscale=:log10, minorgrid=true)
	xlabel!("Closest elec-elec distance")
	ylabel!("Kinetic energy")

	peee = plot(walkers_ee, pel, seriestype=:scatter, label="")
	plot!(xscale=:log10, minorgrid=true)
	xlabel!("Closest elec-elec distance")
	ylabel!("Potential energy")

	# pgfplotsx()
	plot(eea, keea, peea, eee, keee, peee, layout=(2, 3), size=(1000, 500),
	bottom_margin = 10pt, right_margin = 10pt, left_margin=10pt)
	
end

# ╔═╡ 39376428-3d1d-48e2-8ca9-34120e76266c
savefig("det-cusp.pdf")  

# ╔═╡ Cell order:
# ╠═44220f0a-73c3-11ed-2cd4-4d9cd3f98b6a
# ╠═dafcbf36-a777-46b6-a710-838167e452c6
# ╠═e62bd613-cde1-43e3-848c-872e7d1a3f5a
# ╠═186d7956-4ea8-44b0-b753-971d2d40c36e
# ╠═8c3a1b22-c2d2-437a-86b8-0abe8d5fcee4
# ╠═39376428-3d1d-48e2-8ca9-34120e76266c
