using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.RegionIndices
import Sbplib.SbpOperators.BoundaryOperator

@testset "boundary_restriction" begin
	stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order = 4)
	e_closure = parse_stencil(stencil_set["e"]["closure"])
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))

    @testset "boundary_restriction" begin
        @testset "1D" begin
            e_l = boundary_restriction(g_1D,e_closure,CartesianBoundary{1,Lower}())
            @test e_l == boundary_restriction(g_1D,stencil_set,CartesianBoundary{1,Lower}())
            @test e_l == BoundaryOperator(g_1D,Stencil{Float64}(e_closure),Lower())
            @test e_l isa BoundaryOperator{T,Lower} where T
            @test e_l isa LazyTensor{T,0,1} where T

            e_r = boundary_restriction(g_1D,e_closure,CartesianBoundary{1,Upper}())
            @test e_r == boundary_restriction(g_1D,stencil_set,CartesianBoundary{1,Upper}())
            @test e_r == BoundaryOperator(g_1D,Stencil{Float64}(e_closure),Upper())
            @test e_r isa BoundaryOperator{T,Upper} where T
            @test e_r isa LazyTensor{T,0,1} where T
        end

        @testset "2D" begin
            e_w = boundary_restriction(g_2D,e_closure,CartesianBoundary{1,Upper}())
            @test e_w == boundary_restriction(g_2D,stencil_set,CartesianBoundary{1,Upper}())
            @test e_w isa InflatedTensor
            @test e_w isa LazyTensor{T,1,2} where T
        end
    end

    @testset "Application" begin
        @testset "1D" begin
            e_l, e_r = boundary_restriction.(Ref(g_1D), Ref(e_closure), boundary_identifiers(g_1D))
            v = evalOn(g_1D,x->1+x^2)
            u = fill(3.124)

            @test (e_l*v)[] == v[1]
            @test (e_r*v)[] == v[end]
            @test (e_r*v)[1] == v[end]
        end

        @testset "2D" begin
            e_w, e_e, e_s, e_n = boundary_restriction.(Ref(g_2D), Ref(e_closure), boundary_identifiers(g_2D))
            v = rand(11, 15)
            u = fill(3.124)

            @test e_w*v == v[1,:]
            @test e_e*v == v[end,:]
            @test e_s*v == v[:,1]
            @test e_n*v == v[:,end]
       end
    end
end
