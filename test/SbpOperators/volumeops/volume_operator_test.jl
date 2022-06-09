using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.RegionIndices
using Sbplib.LazyTensors

import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.VolumeOperator
import Sbplib.SbpOperators.odd
import Sbplib.SbpOperators.even

# REVIEW: Remove the commented tests for 2D (it is tested in the user code), but
# change testset Regions and testset Inferred, to test the 1D operators.

@testset "VolumeOperator" begin
    inner_stencil = CenteredStencil(1/4, 2/4, 1/4)
    closure_stencils = (Stencil(1/2, 1/2; center=1), Stencil(0.,1.; center=2))
    g = EquidistantGrid(11,0.,1.)
    @testset "Constructors" begin
        op = VolumeOperator(inner_stencil,closure_stencils,(11,),even)
        @test op == VolumeOperator(g,inner_stencil,closure_stencils,even)
        @test op isa LazyTensor{T,1,1} where T
    end

    @testset "Sizes" begin
        op = VolumeOperator(g,inner_stencil,closure_stencils,even)
        @test range_size(op) == domain_size(op) == size(g)
    end

    # op_x = volume_operator(g_2D,inner_stencil,closure_stencils,even,1)
    # op_y = volume_operator(g_2D,inner_stencil,closure_stencils,odd,2)
    # v = zeros(size(g_2D))
    # Nx = size(g_2D)[1]
    # Ny = size(g_2D)[2]
    # for i = 1:Nx
    #     v[i,:] .= i
    # end
    # rx = copy(v)
    # rx[1,:] .= 1.5
    # rx[Nx,:] .= (2*Nx-1)/2
    # ry = copy(v)
    # ry[:,Ny-1:Ny] = -v[:,Ny-1:Ny]

    # @testset "Application" begin
    #     @test op_x*v ≈ rx rtol = 1e-14
    #     @test op_y*v ≈ ry rtol = 1e-14

    #     @test (op_x*rand(ComplexF64,size(g_2D)))[2,2] isa ComplexF64
    # end

    # @testset "Regions" begin
    #     @test (op_x*v)[Index(1,Lower),Index(3,Interior)] ≈ rx[1,3] rtol = 1e-14
    #     @test (op_x*v)[Index(2,Lower),Index(3,Interior)] ≈ rx[2,3] rtol = 1e-14
    #     @test (op_x*v)[Index(6,Interior),Index(3,Interior)] ≈ rx[6,3] rtol = 1e-14
    #     @test (op_x*v)[Index(10,Upper),Index(3,Interior)] ≈ rx[10,3] rtol = 1e-14
    #     @test (op_x*v)[Index(11,Upper),Index(3,Interior)] ≈ rx[11,3] rtol = 1e-14

    #     @test_throws BoundsError (op_x*v)[Index(3,Lower),Index(3,Interior)]
    #     @test_throws BoundsError (op_x*v)[Index(9,Upper),Index(3,Interior)]

    #     @test (op_y*v)[Index(3,Interior),Index(1,Lower)] ≈ ry[3,1] rtol = 1e-14
    #     @test (op_y*v)[Index(3,Interior),Index(2,Lower)] ≈ ry[3,2] rtol = 1e-14
    #     @test (op_y*v)[Index(3,Interior),Index(6,Interior)] ≈ ry[3,6] rtol = 1e-14
    #     @test (op_y*v)[Index(3,Interior),Index(11,Upper)] ≈ ry[3,11] rtol = 1e-14
    #     @test (op_y*v)[Index(3,Interior),Index(12,Upper)] ≈ ry[3,12] rtol = 1e-14

    #     @test_throws BoundsError (op_y*v)[Index(3,Interior),Index(10,Upper)]
    #     @test_throws BoundsError (op_y*v)[Index(3,Interior),Index(3,Lower)]
    # end

    # @testset "Inferred" begin
    #     @test_skip @inferred apply(op_x, v,1,1)
    #     @inferred apply(op_x, v, Index(1,Lower),Index(1,Lower))
    #     @inferred apply(op_x, v, Index(6,Interior),Index(1,Lower))
    #     @inferred apply(op_x, v, Index(11,Upper),Index(1,Lower))
    #     @test_skip @inferred apply(op_y, v,1,1)
    #     @inferred apply(op_y, v, Index(1,Lower),Index(1,Lower))
    #     @inferred apply(op_y, v, Index(1,Lower),Index(6,Interior))
    #     @inferred apply(op_y, v, Index(1,Lower),Index(11,Upper))
    # end
end
