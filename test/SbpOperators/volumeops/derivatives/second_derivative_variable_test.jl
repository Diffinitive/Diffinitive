using Test

using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.SbpOperators
using Sbplib.SbpOperators: NestedStencil, CenteredNestedStencil



@testset "SecondDerivativeVariable" begin
    g = EquidistantGrid(11, 0., 1.)
    c = [  1,  3,  6, 10, 15, 21, 28, 36, 45, 55, 66]

    interior_stencil = CenteredNestedStencil((-1/2,  -1/2, 0.),( 1/2,     1.,  1/2),(   0.,  -1/2, -1/2))
    closure_stencils = [
        NestedStencil(( 1/2,  1/2, 0.),(-1/2, -1/2,  0.), center = 1),
        NestedStencil((-1/2, -1/2, 0.),( 1/2,   1., 1/2), center = 2),
    ]

    @testset "Constructors" begin
        @test SecondDerivativeVariable(interior_stencil,Tuple(closure_stencils), (4,), c) isa TensorMapping
        @test SecondDerivativeVariable(g, c, interior_stencil, closure_stencils) isa TensorMapping
    end

    @testset "sizes" begin
        D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils)
        @test closure_size(D₂ᶜ) == 2
        @test range_size(D₂ᶜ) == (11,)
        @test domain_size(D₂ᶜ) == (11,)
    end

    @testset "application" begin
        # @test D₂(c)*v =
    end
end

