using Test

using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.SbpOperators
using Sbplib.SbpOperators: NestedStencil, CenteredNestedStencil



@testset "SecondDerivativeVariable" begin
    g = EquidistantGrid(11, 0., 1.)
    c = [  1,  3,  6, 10, 15, 21, 28, 36, 45, 55, 66]

    interior_stencil = CenteredNestedStencil((1/2, 1/2, 0.),(-1/2, -1., -1/2),( 0., 1/2, 1/2))
    closure_stencils = [
        NestedStencil(( 2.,  -1., 0.),(-3., 1.,  0.), (1., 0., 0.), center = 1),
    ]

    @testset "1D" begin
        @testset "Constructors" begin
            @test SecondDerivativeVariable(interior_stencil,Tuple(closure_stencils), (4,), c) isa TensorMapping
            @test SecondDerivativeVariable(g, c, interior_stencil, closure_stencils) isa TensorMapping
        end

        @testset "sizes" begin
            D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils)
            @test closure_size(D₂ᶜ) == 1
            @test range_size(D₂ᶜ) == (11,)
            @test domain_size(D₂ᶜ) == (11,)
        end

        @testset "application" begin

            function apply_to_functions(;v,c)
                g = EquidistantGrid(11, 0., 10.) # h = 1
                c̄ = evalOn(g,c)
                v̄ = evalOn(g,v)

                D₂ᶜ = SecondDerivativeVariable(g, c̄, interior_stencil, closure_stencils)
                return D₂ᶜ*v̄
            end

            @test apply_to_functions(v=x->1., c=x->-1.) == zeros(11)
            @test apply_to_functions(v=x->1., c=x->-x) == zeros(11)
            @test apply_to_functions(v=x->x, c=x-> 1.) == zeros(11)
            @test apply_to_functions(v=x->x, c=x->-x) == -ones(11)
        end
    end
end

