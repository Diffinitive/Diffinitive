using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
using Diffinitive.LazyTensors
using Diffinitive.RegionIndices
import Diffinitive.SbpOperators.BoundaryOperator

@testset "normal_derivative" begin
    g_1D = equidistant_grid(0.0, 1.0, 11)
    g_2D = equidistant_grid((0.0, 0.0), (1.0,1.0), 11, 12)
    @testset "normal_derivative" begin
    	stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            d_l = normal_derivative(g_1D, stencil_set, LowerBoundary())
            @test d_l == normal_derivative(g_1D, stencil_set, LowerBoundary())
            @test d_l isa BoundaryOperator{T,LowerBoundary} where T
            @test d_l isa LazyTensor{T,0,1} where T
        end
        @testset "2D" begin
            d_w = normal_derivative(g_2D, stencil_set, CartesianBoundary{1,LowerBoundary}())
            d_n = normal_derivative(g_2D, stencil_set, CartesianBoundary{2,UpperBoundary}())
            Ix = IdentityTensor{Float64}((size(g_2D)[1],))
            Iy = IdentityTensor{Float64}((size(g_2D)[2],))
            d_l = normal_derivative(g_2D.grids[1], stencil_set, LowerBoundary())
            d_r = normal_derivative(g_2D.grids[2], stencil_set, UpperBoundary())
            @test d_w == normal_derivative(g_2D, stencil_set, CartesianBoundary{1,LowerBoundary}())
            @test d_w ==  d_l⊗Iy
            @test d_n ==  Ix⊗d_r
            @test d_w isa LazyTensor{T,1,2} where T
            @test d_n isa LazyTensor{T,1,2} where T
        end
    end
    @testset "Accuracy" begin
        v = eval_on(g_2D, (x,y)-> x^2 + (y-1)^2 + x*y)
        v∂x = eval_on(g_2D, (x,y)-> 2*x + y)
        v∂y = eval_on(g_2D, (x,y)-> 2*(y-1) + x)
        # TODO: Test for higher order polynomials?
        @testset "2nd order" begin
        	stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
            d_w, d_e, d_s, d_n = normal_derivative.(Ref(g_2D), Ref(stencil_set), boundary_identifiers(g_2D))

            @test d_w*v ≈ -v∂x[1,:] atol = 1e-13
            @test d_e*v ≈ v∂x[end,:] atol = 1e-13
            @test d_s*v ≈ -v∂y[:,1] atol = 1e-13
            @test d_n*v ≈ v∂y[:,end] atol = 1e-13
        end

        @testset "4th order" begin
            stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
            d_w, d_e, d_s, d_n = normal_derivative.(Ref(g_2D), Ref(stencil_set), boundary_identifiers(g_2D))
            
            @test d_w*v ≈ -v∂x[1,:] atol = 1e-13
            @test d_e*v ≈ v∂x[end,:] atol = 1e-13
            @test d_s*v ≈ -v∂y[:,1] atol = 1e-13
            @test d_n*v ≈ v∂y[:,end] atol = 1e-13
        end
    end
end
