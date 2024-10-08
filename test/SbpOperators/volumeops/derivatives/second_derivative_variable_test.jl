using Test

using Diffinitive.Grids
using Diffinitive.LazyTensors
using Diffinitive.SbpOperators
using Diffinitive.RegionIndices
using Diffinitive.SbpOperators: NestedStencil, CenteredNestedStencil, SecondDerivativeVariable

using LinearAlgebra

@testset "second_derivative_variable" begin
    stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)

    @testset "1D" begin
        g = equidistant_grid(0., 1., 11)
        c = [  1.,  3.,  6., 10., 15., 21., 28., 36., 45., 55., 66.]

        @testset "checking c" begin
            c_short = rand(5)
            c_long = rand(16)
            c_higher_dimension = rand(11,11)

            @test_throws DimensionMismatch("the size (5,) of the coefficient does not match the size (11,) of the grid") second_derivative_variable(g, c_short, stencil_set)
            @test_throws DimensionMismatch("the size (16,) of the coefficient does not match the size (11,) of the grid") second_derivative_variable(g, c_long, stencil_set)
            @test_throws ArgumentError("The coefficient has dimension 2 while the grid is dimension 1") second_derivative_variable(g, c_higher_dimension, stencil_set)
        end

        @testset "application" begin
            function apply_to_functions(; v, c)
                g = equidistant_grid(0., 10., 11) # h = 1
                c̄ = eval_on(g,c)
                v̄ = eval_on(g,v)

                D₂ᶜ = second_derivative_variable(g, c̄, stencil_set)
                return D₂ᶜ*v̄
            end

            @test apply_to_functions(v=x->1.,  c=x-> -1.) == zeros(11)
            @test apply_to_functions(v=x->1.,  c=x-> -x ) == zeros(11)
            @test apply_to_functions(v=x->x,   c=x->  1.) == zeros(11)
            @test apply_to_functions(v=x->x,   c=x-> -x ) == -ones(11)
            @test apply_to_functions(v=x->x^2, c=x->  1.) == 2ones(11)
        end
    end

    @testset "2D" begin
        g = equidistant_grid((0.,0.), (10.,8.), 11, 9) # h = 1
        c = eval_on(g, (x,y)->x+y)

        @testset "application" begin
            function apply_to_functions(dir; v, c)
                g = equidistant_grid((0.,0.), (10.,8.), 11, 9) # h = 1
                c̄ = eval_on(g,c)
                v̄ = eval_on(g,v)

                D₂ᶜ = second_derivative_variable(g, c̄, stencil_set, dir)
                return D₂ᶜ*v̄
            end

            # x-direction
            @test apply_to_functions(1,v=(x,y)->1.,  c=(x,y)-> -1.) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->1.,  c=(x,y)->- x ) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->x,   c=(x,y)->  1.) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->x,   c=(x,y)-> -x ) == -ones(11,9)
            @test apply_to_functions(1,v=(x,y)->x^2, c=(x,y)->  1.) == 2ones(11,9)

            @test apply_to_functions(1,v=(x,y)->1.,  c=(x,y)->- y ) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->y,   c=(x,y)->  1.) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->y,   c=(x,y)-> -y ) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->y^2, c=(x,y)->  1.) == zeros(11,9)

            # y-direction
            @test apply_to_functions(2,v=(x,y)->1.,  c=(x,y)-> -1.) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->1.,  c=(x,y)->- y ) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->y,   c=(x,y)->  1.) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->y,   c=(x,y)-> -y ) == -ones(11,9)
            @test apply_to_functions(2,v=(x,y)->y^2, c=(x,y)->  1.) == 2ones(11,9)

            @test apply_to_functions(2,v=(x,y)->1.,  c=(x,y)->- x ) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->x,   c=(x,y)->  1.) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->x,   c=(x,y)-> -x ) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->x^2, c=(x,y)->  1.) == zeros(11,9)


            @testset "standard diagonal operators" begin
                c(x,y) = exp(x) + exp(1.5(1-y))
                v(x,y) = sin(x) + cos(1.5(1-y))

                Dxv(x,y) = cos(x)*exp(x) - (exp(x) + exp(1.5 - 1.5y))*sin(x)
                Dyv(x,y) = -1.5(1.5exp(x) + 1.5exp(1.5 - 1.5y))*cos(1.5 - 1.5y) - 2.25exp(1.5 - 1.5y)*sin(1.5 - 1.5y)

                g₁ = equidistant_grid((0.,0.), (1.,2.), 60, 67)
                g₂ = refine(g₁,2)

                c̄₁ = eval_on(g₁, c)
                c̄₂ = eval_on(g₂, c)

                v̄₁ = eval_on(g₁, v)
                v̄₂ = eval_on(g₂, v)


                function convergence_rate_estimate(stencil_set, dir, Dv_true)
                    D₁ = second_derivative_variable(g₁, c̄₁, stencil_set, dir)
                    D₂ = second_derivative_variable(g₂, c̄₂, stencil_set, dir)

                    Dv̄₁ = D₁*v̄₁
                    Dv̄₂ = D₂*v̄₂

                    Dv₁ = eval_on(g₁,Dv_true)
                    Dv₂ = eval_on(g₂,Dv_true)

                    e₁ = norm(Dv̄₁ - Dv₁)/norm(Dv₁)
                    e₂ = norm(Dv̄₂ - Dv₂)/norm(Dv₂)

                    return log2(e₁/e₂)
                end

                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order = 2)
                @test convergence_rate_estimate(stencil_set, 1, Dxv) ≈ 1.5 rtol = 1e-1
                @test convergence_rate_estimate(stencil_set, 2, Dyv) ≈ 1.5 rtol = 1e-1

                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order = 4)
                @test convergence_rate_estimate(stencil_set, 1, Dxv) ≈ 2.5 rtol = 1e-1
                @test convergence_rate_estimate(stencil_set, 2, Dyv) ≈ 2.5 rtol = 2e-1
            end
        end
    end
end


@testset "SecondDerivativeVariable" begin
    interior_stencil = CenteredNestedStencil((1/2, 1/2, 0.),(-1/2, -1., -1/2),( 0., 1/2, 1/2))
    closure_stencils = (
        NestedStencil(( 2.,  -1., 0.),(-3., 1.,  0.), (1., 0., 0.), center = 1),
    )

    @testset "1D" begin
        c = [  1.,  3.,  6., 10., 15., 21., 28., 36., 45., 55., 66.]
        @testset "Constructors" begin
            @test SecondDerivativeVariable(c, interior_stencil, closure_stencils, 1) isa LazyTensor

            D₂ᶜ = SecondDerivativeVariable(c, interior_stencil, closure_stencils, 1)
            @test range_dim(D₂ᶜ) == 1
            @test domain_dim(D₂ᶜ) == 1

        end

        @testset "sizes" begin
            D₂ᶜ = SecondDerivativeVariable(c, interior_stencil, closure_stencils, 1)
            @test closure_size(D₂ᶜ) == 1
            @test range_size(D₂ᶜ) == (11,)
            @test domain_size(D₂ᶜ) == (11,)
        end

        @testset "application" begin

            function apply_to_functions(; v, c)
                g = equidistant_grid(0., 10., 11) # h = 1
                c̄ = eval_on(g,c)
                v̄ = eval_on(g,v)

                D₂ᶜ = SecondDerivativeVariable(c̄, interior_stencil, closure_stencils, 1)
                return D₂ᶜ*v̄
            end

            @test apply_to_functions(v=x->1.,  c=x-> -1.) == zeros(11)
            @test apply_to_functions(v=x->1.,  c=x-> -x ) == zeros(11)
            @test apply_to_functions(v=x->x,   c=x->  1.) == zeros(11)
            @test apply_to_functions(v=x->x,   c=x-> -x ) == -ones(11)
            @test apply_to_functions(v=x->x^2, c=x->  1.) == 2ones(11)
        end

        @testset "type stability" begin
            g = equidistant_grid(0., 10., 11) # h = 1
            c̄ = eval_on(g,x-> -1)
            v̄ = eval_on(g,x->1.)

            D₂ᶜ = SecondDerivativeVariable(c̄, interior_stencil, closure_stencils, 1)

            @inferred SbpOperators.apply_lower(D₂ᶜ, v̄, 1)
            @inferred SbpOperators.apply_interior(D₂ᶜ, v̄, 5)
            @inferred SbpOperators.apply_upper(D₂ᶜ, v̄, 11)
            @inferred (D₂ᶜ*v̄)[Index(1,Lower)]
        end
    end

    @testset "2D" begin
        g = equidistant_grid((0.,0.), (10.,8.), 11, 9) # h = 1
        c = eval_on(g, (x,y)->x+y)
        @testset "Constructors" begin
            @test SecondDerivativeVariable(c, interior_stencil, closure_stencils, 1) isa LazyTensor
            @test SecondDerivativeVariable(c, interior_stencil, closure_stencils, 2) isa LazyTensor

            D₂ᶜ = SecondDerivativeVariable(c, interior_stencil, closure_stencils, 1)
            @test range_dim(D₂ᶜ) == 2
            @test domain_dim(D₂ᶜ) == 2
        end

        @testset "sizes" begin
            D₂ᶜ = SecondDerivativeVariable(c, interior_stencil, closure_stencils, 1)
            @test range_size(D₂ᶜ) == (11,9)
            @test domain_size(D₂ᶜ) == (11,9)
            @test closure_size(D₂ᶜ) == 1

            D₂ᶜ = SecondDerivativeVariable(c, interior_stencil, closure_stencils, 2)
            @test range_size(D₂ᶜ) == (11,9)
            @test domain_size(D₂ᶜ) == (11,9)
            @test closure_size(D₂ᶜ) == 1
        end

        @testset "application" begin
            function apply_to_functions(dir; v, c)
                g = equidistant_grid((0.,0.), (10.,8.), 11, 9) # h = 1
                c̄ = eval_on(g,c)
                v̄ = eval_on(g,v)

                D₂ᶜ = SecondDerivativeVariable(c̄, interior_stencil, closure_stencils, dir)
                return D₂ᶜ*v̄
            end

            # x-direction
            @test apply_to_functions(1,v=(x,y)->1.,  c=(x,y)-> -1.) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->1.,  c=(x,y)->- x ) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->x,   c=(x,y)->  1.) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->x,   c=(x,y)-> -x ) == -ones(11,9)
            @test apply_to_functions(1,v=(x,y)->x^2, c=(x,y)->  1.) == 2ones(11,9)

            @test apply_to_functions(1,v=(x,y)->1.,  c=(x,y)->- y ) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->y,   c=(x,y)->  1.) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->y,   c=(x,y)-> -y ) == zeros(11,9)
            @test apply_to_functions(1,v=(x,y)->y^2, c=(x,y)->  1.) == zeros(11,9)

            # y-direction
            @test apply_to_functions(2,v=(x,y)->1.,  c=(x,y)-> -1.) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->1.,  c=(x,y)->- y ) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->y,   c=(x,y)->  1.) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->y,   c=(x,y)-> -y ) == -ones(11,9)
            @test apply_to_functions(2,v=(x,y)->y^2, c=(x,y)->  1.) == 2ones(11,9)

            @test apply_to_functions(2,v=(x,y)->1.,  c=(x,y)->- x ) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->x,   c=(x,y)->  1.) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->x,   c=(x,y)-> -x ) == zeros(11,9)
            @test apply_to_functions(2,v=(x,y)->x^2, c=(x,y)->  1.) == zeros(11,9)
        end
    end
end

