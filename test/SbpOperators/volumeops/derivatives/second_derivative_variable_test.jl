using Test

using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.SbpOperators
using Sbplib.SbpOperators: NestedStencil, CenteredNestedStencil



@testset "SecondDerivativeVariable" begin
    g = EquidistantGrid(11, 0., 1.)

    interior_stencil = CenteredNestedStencil((-1/2,  -1/2, 0.),( 1/2,     1.,  1/2),(   0.,  -1/2, -1/2))
    closure_stencils = [
        NestedStencil(( 1/2,  1/2, 0.),(-1/2, -1/2,  0.), center = 1),
        NestedStencil((-1/2, -1/2, 0.),( 1/2,   1., 1/2), center = 2),
    ]

    @test SecondDerivativeVariable(interior_stencil,Tuple(closure_stencils), (4,)) isa TensorMapping
    @test SecondDerivativeVariable(g, interior_stencil, closure_stencils) isa TensorMapping
end

