using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.RegionIndices
using Sbplib.LazyTensors

import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.VolumeOperator
import Sbplib.SbpOperators.odd
import Sbplib.SbpOperators.even


@testset "VolumeOperator" begin
    inner_stencil = CenteredStencil(1/4, 2/4, 1/4)
    closure_stencils = (Stencil(1/2, 1/2; center=1), Stencil(2.,1.; center=2))
    g = equidistant_grid(0.,1., 11)

    @testset "Constructors" begin
        op = VolumeOperator(inner_stencil,closure_stencils,(11,),even)
        @test op == VolumeOperator(g,inner_stencil,closure_stencils,even)
        @test op isa LazyTensor{T,1,1} where T
    end

    @testset "Sizes" begin
        op = VolumeOperator(g,inner_stencil,closure_stencils,even)
        @test range_size(op) == domain_size(op) == size(g)
    end


    op_even = VolumeOperator(g, inner_stencil, closure_stencils, even)
    op_odd =  VolumeOperator(g, inner_stencil, closure_stencils, odd)

    N = size(g)[1]
    v = rand(N)

    r_even = copy(v)
    r_odd  = copy(v)

    r_even[1] = (v[1] + v[2])/2
    r_odd[1]  = (v[1] + v[2])/2

    r_even[2] = 2v[1] + v[2]
    r_odd[2]  = 2v[1] + v[2]

    for i ∈ 3:N-2
        r_even[i] = (v[i-1] + 2v[i] + v[i+1])/4
        r_odd[i]  = (v[i-1] + 2v[i] + v[i+1])/4
    end

    r_even[N-1] =  v[N-1] + 2v[N]
    r_odd[N-1]  = -v[N-1] - 2v[N]

    r_even[N] =  (v[N-1] + v[N])/2
    r_odd[N]  = -(v[N-1] + v[N])/2


    @testset "Application" begin
        @test op_even*v ≈ r_even
        @test op_odd*v  ≈ r_odd

        @test (op_even*rand(ComplexF64,size(g)))[2] isa ComplexF64
    end

    @testset "Regions" begin
        @test (op_even*v)[Index(1,Lower)]    ≈ r_even[1]
        @test (op_even*v)[Index(2,Lower)]    ≈ r_even[2]
        @test (op_even*v)[Index(6,Interior)] ≈ r_even[6]
        @test (op_even*v)[Index(10,Upper)]   ≈ r_even[10]
        @test (op_even*v)[Index(11,Upper)]   ≈ r_even[11]

        @test_throws BoundsError (op_even*v)[Index(3,Lower)]
        @test_throws BoundsError (op_even*v)[Index(9,Upper)]
    end

    @testset "Inferred" begin
        @inferred apply(op_even, v, 1)
        @inferred apply(op_even, v, Index(1,Lower))
        @inferred apply(op_even, v, Index(6,Interior))
        @inferred apply(op_even, v, Index(11,Upper))
    end
end
