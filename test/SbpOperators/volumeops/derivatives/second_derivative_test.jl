using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

import Sbplib.SbpOperators.VolumeOperator

@testset "SecondDerivative" begin
    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    stencil_set = read_stencil_set(operator_path; order=4)
    inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
    Lx = 3.5
    Ly = 3.
    g_1D = EquidistantGrid(121, 0.0, Lx)
    g_2D = EquidistantGrid((121,123), (0.0, 0.0), (Lx, Ly))

    @testset "Constructors" begin
        @testset "1D" begin
            Dₓₓ = second_derivative(g_1D,inner_stencil,closure_stencils,1)
            @test Dₓₓ == second_derivative(g_1D,inner_stencil,closure_stencils)
            @test Dₓₓ == second_derivative(g_1D,stencil_set,1)
            @test Dₓₓ isa VolumeOperator
        end
        @testset "2D" begin
            Dₓₓ = second_derivative(g_2D,inner_stencil,closure_stencils,1)
            D2 = second_derivative(g_1D,inner_stencil,closure_stencils)
            I = IdentityMapping{Float64}(size(g_2D)[2])
            @test Dₓₓ == D2⊗I
            @test Dₓₓ == second_derivative(g_2D,stencil_set,1)
            @test Dₓₓ isa TensorMapping{T,2,2} where T
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
                monomials = (monomials...,evalOn(g_1D,f_i))
            end
            v = evalOn(g_1D,x -> sin(x))
            vₓₓ = evalOn(g_1D,x -> -sin(x))

            # 2nd order interior stencil, 1nd order boundary stencil,
            # implies that L*v should be exact for monomials up to order 2.
            @testset "2nd order" begin
                stencil_set = read_stencil_set(operator_path; order=2)
                inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
			    closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
                Dₓₓ = second_derivative(g_1D,inner_stencil,closure_stencils)
                @test Dₓₓ*monomials[1] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[2] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[3] ≈ monomials[1] atol = 5e-10
                @test Dₓₓ*v ≈ vₓₓ rtol = 5e-2 norm = l2
            end

            # 4th order interior stencil, 2nd order boundary stencil,
            # implies that L*v should be exact for monomials up to order 3.
            @testset "4th order" begin
                stencil_set = read_stencil_set(operator_path; order=4)
                inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
			    closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
                Dₓₓ = second_derivative(g_1D,inner_stencil,closure_stencils)
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
            l2(v) = sqrt(prod(spacing(g_2D))*sum(v.^2));
            binomials = ()
            maxOrder = 4;
            for i = 0:maxOrder-1
                f_i(x,y) = 1/factorial(i)*y^i + x^i
                binomials = (binomials...,evalOn(g_2D,f_i))
            end
            v = evalOn(g_2D, (x,y) -> sin(x)+cos(y))
            v_yy = evalOn(g_2D,(x,y) -> -cos(y))

            # 2nd order interior stencil, 1st order boundary stencil,
            # implies that L*v should be exact for binomials up to order 2.
            @testset "2nd order" begin
                stencil_set = read_stencil_set(operator_path; order=2)
                inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
                closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
                Dyy = second_derivative(g_2D,inner_stencil,closure_stencils,2)
                @test Dyy*binomials[1] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[2] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[3] ≈ evalOn(g_2D,(x,y)->1.) atol = 5e-9
                @test Dyy*v ≈ v_yy rtol = 5e-2 norm = l2
            end

            # 4th order interior stencil, 2nd order boundary stencil,
            # implies that L*v should be exact for binomials up to order 3.
            @testset "4th order" begin
                stencil_set = read_stencil_set(operator_path; order=4)
                inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
                closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
                Dyy = second_derivative(g_2D,inner_stencil,closure_stencils,2)
                # NOTE: high tolerances for checking the "exact" differentiation
                # due to accumulation of round-off errors/cancellation errors?
                @test Dyy*binomials[1] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[2] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[3] ≈ evalOn(g_2D,(x,y)->1.) atol = 5e-9
                @test Dyy*binomials[4] ≈ evalOn(g_2D,(x,y)->y) atol = 5e-9
                @test Dyy*v ≈ v_yy rtol = 5e-4 norm = l2
            end
        end
    end
end
