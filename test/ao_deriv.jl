using VariationalMC
using VariationalMC.molecule
using Test

@testset "Deriv of Atomic Orbitals" begin
    @testset "Deriv of H with STO-3G" begin
        basis = read_basis("sto-3g")
        H_basis = basis["H"]
        H = Molecule([Atom(1, [0.0, 0.0, 0.0], H_basis)], (1, 0))
        ao_deriv = eval_ao_deriv(H, [0.1, 0.2, -0.3])
        @test size(ao_deriv) == (3, 1)
        answer = [-0.15082577; -0.30165155; 0.45247732;;]
        @test all(isapprox.(ao_deriv, answer, atol = 1e-7))
    end
    @testset "Deriv of Li₂ with 6-31G" begin
        basis = read_basis("6-31g")
        Li_basis = basis["Li"]
        Li₂ = Molecule([
            Atom(3, [0.0, 0.0, 0.0], Li_basis),
            Atom(3, [5.0, 0.0, 0.0], Li_basis),
        ])
        ao_deriv = eval_ao_deriv(Li₂, [2.1, 0.2, -0.3])
        answer = [
            -2.67479485e-02, -1.21591672e-02, -7.55057336e-03,
            -1.74626725e-02, -6.03191689e-03, 9.04787533e-03,
            1.29461869e-02, -5.72745705e-04, 8.59118557e-04,
            2.81947334e-03, 2.52860731e-02, 9.02996863e-03,
            -1.44100986e-02, 3.01230352e-03, -4.51845528e-03,
            6.48775462e-03, 6.84964638e-04, -1.02744696e-03,
        ]
        @test length(answer) == length(ao_deriv[1, :])
        @test all(isapprox.(ao_deriv[1, :], answer, atol = 1e-7))
    end
end
