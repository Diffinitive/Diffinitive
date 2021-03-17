using Test
using Sbplib.SbpOperators
import Sbplib.SbpOperators.Stencil

@testset "Stencil" begin
    s = Stencil((-2,2), (1.,2.,2.,3.,4.))
    @test s isa Stencil{Float64, 5}

    @test eltype(s) == Float64
    @test SbpOperators.scale(s, 2) == Stencil((-2,2), (2.,4.,4.,6.,8.))

    @test Stencil(1,2,3,4; center=1) == Stencil((0, 3),(1,2,3,4))
    @test Stencil(1,2,3,4; center=2) == Stencil((-1, 2),(1,2,3,4))
    @test Stencil(1,2,3,4; center=4) == Stencil((-3, 0),(1,2,3,4))

    @test CenteredStencil(1,2,3,4,5) == Stencil((-2, 2), (1,2,3,4,5))
    @test_throws ArgumentError CenteredStencil(1,2,3,4)
end
