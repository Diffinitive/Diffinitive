using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.RegionIndices
using Sbplib.LazyTensors

import Sbplib.SbpOperators.BoundaryOperator

@testset "boundary_restriction" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))

    @testset "boundary_restriction" begin
        @testset "1D" begin
            e_l = boundary_restriction(g_1D,op.eClosure,Lower())
            @test e_l == boundary_restriction(g_1D,op.eClosure,CartesianBoundary{1,Lower}())
            @test e_l == BoundaryOperator(g_1D,op.eClosure,Lower())
            @test e_l isa BoundaryOperator{T,Lower} where T
            @test e_l isa TensorMapping{T,0,1} where T

            e_r = boundary_restriction(g_1D,op.eClosure,Upper())
            @test e_r == boundary_restriction(g_1D,op.eClosure,CartesianBoundary{1,Upper}())
            @test e_r == BoundaryOperator(g_1D,op.eClosure,Upper())
            @test e_r isa BoundaryOperator{T,Upper} where T
            @test e_r isa TensorMapping{T,0,1} where T
        end

        @testset "2D" begin
            e_w = boundary_restriction(g_2D,op.eClosure,CartesianBoundary{1,Upper}())
            @test e_w isa InflatedTensorMapping
            @test e_w isa TensorMapping{T,1,2} where T
        end
    end

    @testset "Application" begin
        @testset "1D" begin
            e_l = boundary_restriction(g_1D, op.eClosure, CartesianBoundary{1,Lower}())
            e_r = boundary_restriction(g_1D, op.eClosure, CartesianBoundary{1,Upper}())

            v = evalOn(g_1D,x->1+x^2)
            u = fill(3.124)

            @test (e_l*v)[] == v[1]
            @test (e_r*v)[] == v[end]
            @test (e_r*v)[1] == v[end]
        end

        @testset "2D" begin
            e_w = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{1,Lower}())
            e_e = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{1,Upper}())
            e_s = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{2,Lower}())
            e_n = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{2,Upper}())

            v = rand(11, 15)
            u = fill(3.124)

            @test e_w*v == v[1,:]
            @test e_e*v == v[end,:]
            @test e_s*v == v[:,1]
            @test e_n*v == v[:,end]
       end
    end
end
