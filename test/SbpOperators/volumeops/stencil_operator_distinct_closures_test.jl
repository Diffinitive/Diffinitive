using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
using Diffinitive.LazyTensors

import Diffinitive.SbpOperators.Stencil
import Diffinitive.SbpOperators.StencilOperatorDistinctClosures

@testset "StencilOperatorDistinctClosures" begin
    g = equidistant_grid(0., 1., 11)

    lower_closure = (
        Stencil(-1,1, center=1),
        Stencil(-2,2, center=2),
    )

    inner_stencil = Stencil(-3,3, center=1)

    upper_closure = (
        Stencil(4,-4,4, center=1),
        Stencil(5,-5,5, center=2),
        Stencil(6,-6,6, center=3),
    )

    A = StencilOperatorDistinctClosures(g, inner_stencil, lower_closure, upper_closure)
    @test A isa LazyTensor{T,1,1} where T

    @test SbpOperators.lower_closure_size(A) == 2
    @test SbpOperators.upper_closure_size(A) == 3

    @test domain_size(A) == (11,)
    @test range_size(A) == (11,)

    v = rand(11)
    @testset "apply" begin
        # Lower closure
        @test LazyTensors.apply(A, v, 1) ≈ 1*(-v[1] + v[2])
        @test LazyTensors.apply(A, v, 2) ≈ 2*(-v[1] + v[2])

        # Interior
        @test LazyTensors.apply(A, v, 3) ≈ 3*(-v[3] + v[4])
        @test LazyTensors.apply(A, v, 4) ≈ 3*(-v[4] + v[5])
        @test LazyTensors.apply(A, v, 8) ≈ 3*(-v[8] + v[9])

        # Upper closure
        @test LazyTensors.apply(A, v,  9) ≈ 4*(v[9] - v[10] + v[11])
        @test LazyTensors.apply(A, v, 10) ≈ 5*(v[9] - v[10] + v[11])
        @test LazyTensors.apply(A, v, 11) ≈ 6*(v[9] - v[10] + v[11])
    end
end
