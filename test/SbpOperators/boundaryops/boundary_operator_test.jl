using Test

using Sbplib.LazyTensors
using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.RegionIndices
import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.BoundaryOperator


@testset "BoundaryOperator" begin
    closure_stencil = Stencil(2.,1.,3.; center = 1)
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))

    @testset "Constructors" begin
        op_l = BoundaryOperator{Lower}(closure_stencil,size(g_1D)[1])
        @test op_l == BoundaryOperator(g_1D,closure_stencil,Lower())
        @test op_l isa LazyTensor{T,0,1} where T

        op_r = BoundaryOperator{Upper}(closure_stencil,size(g_1D)[1])
        @test op_r == BoundaryOperator(g_1D,closure_stencil,Upper())
        @test op_r isa LazyTensor{T,0,1} where T
    end

    op_l = BoundaryOperator(g_1D, closure_stencil, Lower())
    op_r = BoundaryOperator(g_1D, closure_stencil, Upper())

    @testset "Sizes" begin
        @test domain_size(op_l) == (11,)
        @test domain_size(op_r) == (11,)

        @test range_size(op_l) == ()
        @test range_size(op_r) == ()
    end

    @testset "Application" begin
        v = evalOn(g_1D,x->1+x^2)
        u = fill(3.124)
        @test (op_l*v)[] == 2*v[1] + v[2] + 3*v[3]
        @test (op_r*v)[] == 2*v[end] + v[end-1] + 3*v[end-2]
        @test (op_r*v)[1] == 2*v[end] + v[end-1] + 3*v[end-2]
        @test op_l'*u == [2*u[]; u[]; 3*u[]; zeros(8)]
        @test op_r'*u == [zeros(8); 3*u[]; u[]; 2*u[]]

        v = evalOn(g_1D, x->1. +x*im)
        @test (op_l*v)[] isa ComplexF64

        u = fill(1. +im)
        @test (op_l'*u)[1] isa ComplexF64
        @test (op_l'*u)[5] isa ComplexF64
        @test (op_l'*u)[11] isa ComplexF64

        u = fill(3.124)
        @test (op_l'*u)[Index(1,Lower)] == 2*u[]
        @test (op_l'*u)[Index(2,Lower)] == u[]
        @test (op_l'*u)[Index(6,Interior)] == 0
        @test (op_l'*u)[Index(10,Upper)] == 0
        @test (op_l'*u)[Index(11,Upper)] == 0

        @test (op_r'*u)[Index(1,Lower)] == 0
        @test (op_r'*u)[Index(2,Lower)] == 0
        @test (op_r'*u)[Index(6,Interior)] == 0
        @test (op_r'*u)[Index(10,Upper)] == u[]
        @test (op_r'*u)[Index(11,Upper)] == 2*u[]
    end

    @testset "Inferred" begin
        v = ones(Float64, 11)
        u = fill(1.)

        @inferred apply(op_l, v)
        @inferred apply(op_r, v)

        @inferred apply_transpose(op_l, u, 4)
        @inferred apply_transpose(op_l, u, Index(1,Lower))
        @inferred apply_transpose(op_l, u, Index(2,Lower))
        @inferred apply_transpose(op_l, u, Index(6,Interior))
        @inferred apply_transpose(op_l, u, Index(10,Upper))
        @inferred apply_transpose(op_l, u, Index(11,Upper))

        @inferred apply_transpose(op_r, u, 4)
        @inferred apply_transpose(op_r, u, Index(1,Lower))
        @inferred apply_transpose(op_r, u, Index(2,Lower))
        @inferred apply_transpose(op_r, u, Index(6,Interior))
        @inferred apply_transpose(op_r, u, Index(10,Upper))
        @inferred apply_transpose(op_r, u, Index(11,Upper))
    end
end
