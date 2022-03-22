using Test

using Sbplib.SbpOperators
using Sbplib.Grids
# using Sbplib.RegionIndices
using Sbplib.LazyTensors

import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.StencilOperatorDistinctClosures
import Sbplib.SbpOperators.stencil_operator_distinct_closures

@testset "stencil_operator_distinct_closures" begin
    lower_closure = (
        Stencil(-1,1, center=1),
    )

    inner_stencil = Stencil(-2,2, center=1)

    upper_closure = (
        Stencil(-3,3, center=1),
        Stencil(-4,4, center=2),
    )

    g₁ = EquidistantGrid(5, 0., 1.)
    g₂ = EquidistantGrid((5,5), (0.,0.), (1.,1.))
    h = 1/4

    A₁  = stencil_operator_distinct_closures(g₁, inner_stencil, lower_closure, upper_closure, 1)
    A₂¹ = stencil_operator_distinct_closures(g₂, inner_stencil, lower_closure, upper_closure, 1)
    A₂² = stencil_operator_distinct_closures(g₂, inner_stencil, lower_closure, upper_closure, 2)

    v₁ = evalOn(g₁, x->x)

    u = [1., 2., 2., 3., 4.]*h
    @test A₁*v₁ == u

    v₂ = evalOn(g₂, (x,y)-> x + 3y)
    @test A₂¹*v₂ == repeat(u, 1, 5)
    @test A₂²*v₂ == repeat(3u', 5, 1)
end

@testset "StencilOperatorDistinctClosures" begin
    g = EquidistantGrid(11, 0., 1.)

    lower_closure = (
        Stencil(-1,1,0, center=1),
        Stencil(-2,2,0, center=2),
    )

    inner_stencil = Stencil(-3,3, center=1)

    upper_closure = (
        Stencil(4,-4,4, center=1),
        Stencil(0,-5,5, center=2),
        Stencil(0,-6,6, center=3),
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
        @test LazyTensors.apply(A, v, 10) ≈ 5*(     - v[10] + v[11])
        @test LazyTensors.apply(A, v, 11) ≈ 6*(     - v[10] + v[11])
    end
end
