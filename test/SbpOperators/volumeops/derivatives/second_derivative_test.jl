using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

import Sbplib.SbpOperators.VolumeOperator

# TODO: Refactor these test to look more like the tests in first_derivative_test.jl.

@testset "SecondDerivative" begin
    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    stencil_set = read_stencil_set(operator_path; order=4)
    inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
    Lx = 3.5
    Ly = 3.
    g_1D = equidistant_grid(0.0, Lx, 121)
    g_2D = equidistant_grid((0.0, 0.0), (Lx, Ly), 121, 123)

    @testset "Constructors" begin
        @testset "1D" begin
            Dₓₓ = second_derivative(g_1D, stencil_set)
            @test Dₓₓ == second_derivative(g_1D, inner_stencil, closure_stencils)
            @test Dₓₓ isa LazyTensor{Float64,1,1}
        end
        @testset "2D" begin
            Dₓₓ = second_derivative(g_2D,stencil_set,1)
            @test Dₓₓ isa LazyTensor{Float64,2,2}
        end
    end

    # Exact differentiation is measured point-wise. In other cases
    # the error is measured in the l2-norm.
    @testset "Accuracy" begin
        @testset "1D" begin
            l2(v) = sqrt(spacing(g_1D)[1]*sum(v.^2));
            monomials = ()
            maxOrder = 4;
            for i = 0:maxOrder-1
                f_i(x) = 1/factorial(i)*x^i
                monomials = (monomials...,eval_on(g_1D,f_i))
            end
            v = eval_on(g_1D,x -> sin(x))
            vₓₓ = eval_on(g_1D,x -> -sin(x))

            # 2nd order interior stencil, 1nd order boundary stencil,
            # implies that L*v should be exact for monomials up to order 2.
            @testset "2nd order" begin
                stencil_set = read_stencil_set(operator_path; order=2)
                Dₓₓ = second_derivative(g_1D,stencil_set)
                @test Dₓₓ*monomials[1] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[2] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[3] ≈ monomials[1] atol = 5e-10
                @test Dₓₓ*v ≈ vₓₓ rtol = 5e-2 norm = l2
            end

            # 4th order interior stencil, 2nd order boundary stencil,
            # implies that L*v should be exact for monomials up to order 3.
            @testset "4th order" begin
                stencil_set = read_stencil_set(operator_path; order=4)
                Dₓₓ = second_derivative(g_1D,stencil_set)
                # NOTE: high tolerances for checking the "exact" differentiation
                # due to accumulation of round-off errors/cancellation errors?
                @test Dₓₓ*monomials[1] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[2] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[3] ≈ monomials[1] atol = 5e-10
                @test Dₓₓ*monomials[4] ≈ monomials[2] atol = 5e-10
                @test Dₓₓ*v ≈ vₓₓ rtol = 5e-4 norm = l2
            end
        end

        @testset "2D" begin
            l2(v) = sqrt(prod(spacing.(g_2D.grids))*sum(v.^2));
            binomials = ()
            maxOrder = 4;
            for i = 0:maxOrder-1
                f_i(x,y) = 1/factorial(i)*y^i + x^i
                binomials = (binomials...,eval_on(g_2D,f_i))
            end
            v = eval_on(g_2D, (x,y) -> sin(x)+cos(y))
            v_yy = eval_on(g_2D,(x,y) -> -cos(y))

            # 2nd order interior stencil, 1st order boundary stencil,
            # implies that L*v should be exact for binomials up to order 2.
            @testset "2nd order" begin
                stencil_set = read_stencil_set(operator_path; order=2)
                Dyy = second_derivative(g_2D,stencil_set,2)
                @test Dyy*binomials[1] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[2] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[3] ≈ eval_on(g_2D,(x,y)->1.) atol = 5e-9
                @test Dyy*v ≈ v_yy rtol = 5e-2 norm = l2
            end

            # 4th order interior stencil, 2nd order boundary stencil,
            # implies that L*v should be exact for binomials up to order 3.
            @testset "4th order" begin
                stencil_set = read_stencil_set(operator_path; order=4)
                Dyy = second_derivative(g_2D,stencil_set,2)
                # NOTE: high tolerances for checking the "exact" differentiation
                # due to accumulation of round-off errors/cancellation errors?
                @test Dyy*binomials[1] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[2] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[3] ≈ eval_on(g_2D,(x,y)->1.) atol = 5e-9
                @test Dyy*binomials[4] ≈ eval_on(g_2D,(x,y)->y) atol = 5e-9
                @test Dyy*v ≈ v_yy rtol = 5e-4 norm = l2
            end
        end
    end
end
