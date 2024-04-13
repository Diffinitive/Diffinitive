using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

@testset "Laplace" begin
    # Default stencils (4th order)
    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    stencil_set = read_stencil_set(operator_path; order=4)
    g_1D = equidistant_grid(0.0, 1., 101)
    g_3D = equidistant_grid((0.0, -1.0, 0.0), (1., 1., 1.), 51, 101, 52)

    @testset "Constructors" begin
        @testset "1D" begin
            @test Laplace(g_1D, stencil_set) == Laplace(laplace(g_1D, stencil_set), stencil_set)
            @test Laplace(g_1D, stencil_set) isa LazyTensor{Float64,1,1}
        end
        @testset "3D" begin
            @test Laplace(g_3D, stencil_set) == Laplace(laplace(g_3D, stencil_set),stencil_set)
            @test Laplace(g_3D, stencil_set) isa LazyTensor{Float64,3,3}
        end
    end

    # Exact differentiation is measured point-wise. In other cases
    # the error is measured in the l2-norm.
    @testset "Accuracy" begin
        l2(v) = sqrt(prod(spacing.(g_3D.grids))*sum(v.^2));
        polynomials = ()
        maxOrder = 4;
        for i = 0:maxOrder-1
            f_i(x,y,z) = 1/factorial(i)*(y^i + x^i + z^i)
            polynomials = (polynomials...,eval_on(g_3D,f_i))
        end
        # v = eval_on(g_3D, (x,y,z) -> sin(x) + cos(y) + exp(z))
        # Δv = eval_on(g_3D,(x,y,z) -> -sin(x) - cos(y) + exp(z))

        v =  eval_on(g_3D, x̄ -> sin(x̄[1]) + cos(x̄[2]) + exp(x̄[3]))
        Δv = eval_on(g_3D, x̄ -> -sin(x̄[1]) - cos(x̄[2]) + exp(x̄[3]))
        @inferred v[1,2,3]

        # 2nd order interior stencil, 1st order boundary stencil,
        # implies that L*v should be exact for binomials up to order 2.
        @testset "2nd order" begin
            stencil_set = read_stencil_set(operator_path; order=2)
            Δ = Laplace(g_3D, stencil_set)
            @test Δ*polynomials[1] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test Δ*polynomials[2] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test Δ*polynomials[3] ≈ polynomials[1] atol = 5e-9
            @test Δ*v ≈ Δv rtol = 5e-2 norm = l2
        end

        # 4th order interior stencil, 2nd order boundary stencil,
        # implies that L*v should be exact for binomials up to order 3.
        @testset "4th order" begin
            stencil_set = read_stencil_set(operator_path; order=4)
            Δ = Laplace(g_3D, stencil_set)
            # NOTE: high tolerances for checking the "exact" differentiation
            # due to accumulation of round-off errors/cancellation errors?
            @test Δ*polynomials[1] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test Δ*polynomials[2] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test Δ*polynomials[3] ≈ polynomials[1] atol = 5e-9
            @test Δ*polynomials[4] ≈ polynomials[2] atol = 5e-9
            @test Δ*v ≈ Δv rtol = 5e-4 norm = l2
        end
    end
end

@testset "laplace" begin
    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    stencil_set = read_stencil_set(operator_path; order=4)
    g_1D = equidistant_grid(0.0, 1., 101)
    g_3D = equidistant_grid((0.0, -1.0, 0.0), (1., 1., 1.), 51, 101, 52)

    @testset "1D" begin
        Δ = laplace(g_1D, stencil_set)
        @test Δ == second_derivative(g_1D, stencil_set)
        @test Δ isa LazyTensor{Float64,1,1}
    end
    @testset "3D" begin
        Δ = laplace(g_3D, stencil_set)
        @test Δ isa LazyTensor{Float64,3,3}
        Dxx = second_derivative(g_3D, stencil_set, 1)
        Dyy = second_derivative(g_3D, stencil_set, 2)
        Dzz = second_derivative(g_3D, stencil_set, 3)
        @test Δ == Dxx + Dyy + Dzz
        @test Δ isa LazyTensor{Float64,3,3}
    end
end

