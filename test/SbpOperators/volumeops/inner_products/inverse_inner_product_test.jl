using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

import Sbplib.SbpOperators.Stencil

@testset "Diagonal-stencil inverse_inner_product" begin
    Lx = π/2.
    Ly = Float64(π)
    g_1D = EquidistantGrid(77, 0.0, Lx)
    g_2D = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))
    @testset "inverse_inner_product" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "0D" begin
            Hi = inverse_inner_product(EquidistantGrid{Float64}(),op.quadratureClosure)
            @test Hi == IdentityMapping{Float64}()
            @test Hi isa TensorMapping{T,0,0} where T
        end
        @testset "1D" begin
            Hi = inverse_inner_product(g_1D, op.quadratureClosure);
            inner_stencil = CenteredStencil(1.)
            closures = ()
            for i = 1:length(op.quadratureClosure)
                closures = (closures...,Stencil(op.quadratureClosure[i].range,1.0./op.quadratureClosure[i].weights))
            end
            @test Hi == inverse_inner_product(g_1D,closures,inner_stencil)
            @test Hi isa TensorMapping{T,1,1} where T
        end
        @testset "2D" begin
            Hi = inverse_inner_product(g_2D,op.quadratureClosure)
            Hi_x = inverse_inner_product(restrict(g_2D,1),op.quadratureClosure)
            Hi_y = inverse_inner_product(restrict(g_2D,2),op.quadratureClosure)
            @test Hi == Hi_x⊗Hi_y
            @test Hi isa TensorMapping{T,2,2} where T
        end
    end

    @testset "Sizes" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            Hi = inverse_inner_product(g_1D,op.quadratureClosure)
            @test domain_size(Hi) == size(g_1D)
            @test range_size(Hi) == size(g_1D)
        end
        @testset "2D" begin
            Hi = inverse_inner_product(g_2D,op.quadratureClosure)
            @test domain_size(Hi) == size(g_2D)
            @test range_size(Hi) == size(g_2D)
        end
    end

    @testset "Accuracy" begin
        @testset "1D" begin
            v = evalOn(g_1D,x->sin(x))
            u = evalOn(g_1D,x->x^3-x^2+1)
            @testset "2nd order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = inner_product(g_1D,op.quadratureClosure)
                Hi = inverse_inner_product(g_1D,op.quadratureClosure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = inner_product(g_1D,op.quadratureClosure)
                Hi = inverse_inner_product(g_1D,op.quadratureClosure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
        end
        @testset "2D" begin
            v = evalOn(g_2D,(x,y)->sin(x)+cos(y))
            u = evalOn(g_2D,(x,y)->x*y + x^5 - sqrt(y))
            @testset "2nd order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = inner_product(g_2D,op.quadratureClosure)
                Hi = inverse_inner_product(g_2D,op.quadratureClosure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = inner_product(g_2D,op.quadratureClosure)
                Hi = inverse_inner_product(g_2D,op.quadratureClosure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
        end
    end
end
