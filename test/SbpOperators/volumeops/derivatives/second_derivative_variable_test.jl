using Test

using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.SbpOperators
using Sbplib.SbpOperators: NestedStencil, CenteredNestedStencil



@testset "SecondDerivativeVariable" begin
    interior_stencil = CenteredNestedStencil((1/2, 1/2, 0.),(-1/2, -1., -1/2),( 0., 1/2, 1/2))
    closure_stencils = [
        NestedStencil(( 2.,  -1., 0.),(-3., 1.,  0.), (1., 0., 0.), center = 1),
    ]

    @testset "1D" begin
        g = EquidistantGrid(11, 0., 1.)
        c = [  1,  3,  6, 10, 15, 21, 28, 36, 45, 55, 66]
        @testset "Constructors" begin
            @test SecondDerivativeVariable(g, c, interior_stencil, closure_stencils) isa TensorMapping

            D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils)
            @test range_dim(D₂ᶜ) == 1
            @test domain_dim(D₂ᶜ) == 1
        end

        @testset "sizes" begin
            D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils)
            @test closure_size(D₂ᶜ) == 1
            @test range_size(D₂ᶜ) == (11,)
            @test domain_size(D₂ᶜ) == (11,)
        end

        @testset "application" begin

            function apply_to_functions(; v, c)
                g = EquidistantGrid(11, 0., 10.) # h = 1
                c̄ = evalOn(g,c)
                v̄ = evalOn(g,v)

                D₂ᶜ = SecondDerivativeVariable(g, c̄, interior_stencil, closure_stencils)
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
        g = EquidistantGrid((11,9), (0.,0.), (10.,8.)) # h = 1
        c = evalOn(g, (x,y)->x+y)
        @testset "Constructors" begin
            @test SecondDerivativeVariable(g, c, interior_stencil, closure_stencils,1) isa TensorMapping
            @test SecondDerivativeVariable(g, c, interior_stencil, closure_stencils,2) isa TensorMapping

            D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils,1)
            @test range_dim(D₂ᶜ) == 2
            @test domain_dim(D₂ᶜ) == 2
        end

        @testset "sizes" begin
            D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils,1)
            @test range_size(D₂ᶜ) == (11,9)
            @test domain_size(D₂ᶜ) == (11,9)
            @test closure_size(D₂ᶜ) == 1

            D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils,2)
            @test range_size(D₂ᶜ) == (11,9)
            @test domain_size(D₂ᶜ) == (11,9)
            @test closure_size(D₂ᶜ) == 1
        end

        @testset "application" begin
            function apply_to_functions(dir; v, c)
                g = EquidistantGrid((11,9), (0.,0.), (10.,8.)) # h = 1
                c̄ = evalOn(g,c)
                v̄ = evalOn(g,v)

                D₂ᶜ = SecondDerivativeVariable(g, c̄, interior_stencil, closure_stencils,dir)
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

