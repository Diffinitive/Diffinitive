using Test

using Sbplib.SbpOperators
using Sbplib.Grids

@testset "normal_derivative" begin
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,12), (0.0, 0.0), (1.0,1.0))
    @testset "normal_derivative" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            d_l = normal_derivative(g_1D, op.dClosure, Lower())
            @test d_l == normal_derivative(g_1D, op.dClosure, CartesianBoundary{1,Lower}())
            @test d_l isa BoundaryOperator{T,Lower} where T
            @test d_l isa TensorMapping{T,0,1} where T
        end
        @testset "2D" begin
            op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
            d_w = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Lower}())
            d_n = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Upper}())
            Ix = IdentityMapping{Float64}((size(g_2D)[1],))
            Iy = IdentityMapping{Float64}((size(g_2D)[2],))
            d_l = normal_derivative(restrict(g_2D,1),op.dClosure,Lower())
            d_r = normal_derivative(restrict(g_2D,2),op.dClosure,Upper())
            @test d_w ==  d_l⊗Iy
            @test d_n ==  Ix⊗d_r
            @test d_w isa TensorMapping{T,1,2} where T
            @test d_n isa TensorMapping{T,1,2} where T
        end
    end
    @testset "Accuracy" begin
        v = evalOn(g_2D, (x,y)-> x^2 + (y-1)^2 + x*y)
        v∂x = evalOn(g_2D, (x,y)-> 2*x + y)
        v∂y = evalOn(g_2D, (x,y)-> 2*(y-1) + x)
        # TODO: Test for higher order polynomials?
        @testset "2nd order" begin
            op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
            d_w = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Lower}())
            d_e = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Upper}())
            d_s = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Lower}())
            d_n = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Upper}())

            @test d_w*v ≈ v∂x[1,:] atol = 1e-13
            @test d_e*v ≈ -v∂x[end,:] atol = 1e-13
            @test d_s*v ≈ v∂y[:,1] atol = 1e-13
            @test d_n*v ≈ -v∂y[:,end] atol = 1e-13
        end

        @testset "4th order" begin
            op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
            d_w = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Lower}())
            d_e = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Upper}())
            d_s = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Lower}())
            d_n = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Upper}())

            @test d_w*v ≈ v∂x[1,:] atol = 1e-13
            @test d_e*v ≈ -v∂x[end,:] atol = 1e-13
            @test d_s*v ≈ v∂y[:,1] atol = 1e-13
            @test d_n*v ≈ -v∂y[:,end] atol = 1e-13
        end
    end
end
