using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

import Sbplib.SbpOperators.ConstantInteriorScalingOperator

@testset "Diagonal-stencil inverse_inner_product" begin
    Lx = π/2.
    Ly = Float64(π)
    g_1D = EquidistantGrid(77, 0.0, Lx)
    g_2D = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))
    @testset "inverse_inner_product" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
        quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
        @testset "0D" begin
            Hi = inverse_inner_product(EquidistantGrid{Float64}(), quadrature_interior, quadrature_closure)
            @test Hi == inverse_inner_product(EquidistantGrid{Float64}(), stencil_set)
            @test Hi == IdentityTensor{Float64}()
            @test Hi isa LazyTensor{T,0,0} where T
        end
        @testset "1D" begin
            Hi = inverse_inner_product(g_1D,  quadrature_interior, quadrature_closure)
            @test Hi == inverse_inner_product(g_1D, stencil_set)
            @test Hi isa ConstantInteriorScalingOperator
            @test Hi isa LazyTensor{T,1,1} where T
        end
        @testset "2D" begin
            Hi = inverse_inner_product(g_2D, quadrature_interior, quadrature_closure)
            Hi_x = inverse_inner_product(restrict(g_2D,1), quadrature_interior, quadrature_closure)
            Hi_y = inverse_inner_product(restrict(g_2D,2), quadrature_interior, quadrature_closure)
            @test Hi == inverse_inner_product(g_2D, stencil_set)
            @test Hi == Hi_x⊗Hi_y
            @test Hi isa LazyTensor{T,2,2} where T
        end
    end

    @testset "Sizes" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
        quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
        @testset "1D" begin
            Hi = inverse_inner_product(g_1D, quadrature_interior, quadrature_closure)
            @test domain_size(Hi) == size(g_1D)
            @test range_size(Hi) == size(g_1D)
        end
        @testset "2D" begin
            Hi = inverse_inner_product(g_2D, quadrature_interior, quadrature_closure)
            @test domain_size(Hi) == size(g_2D)
            @test range_size(Hi) == size(g_2D)
        end
    end

    @testset "Accuracy" begin
        @testset "1D" begin
            v = evalOn(g_1D,x->sin(x))
            u = evalOn(g_1D,x->x^3-x^2+1)
            @testset "2nd order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
                quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
                H = inner_product(g_1D, quadrature_interior, quadrature_closure)
                Hi = inverse_inner_product(g_1D, quadrature_interior, quadrature_closure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
            @testset "4th order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
                quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
                H = inner_product(g_1D, quadrature_interior, quadrature_closure)
                Hi = inverse_inner_product(g_1D, quadrature_interior, quadrature_closure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
        end
        @testset "2D" begin
            v = evalOn(g_2D,(x,y)->sin(x)+cos(y))
            u = evalOn(g_2D,(x,y)->x*y + x^5 - sqrt(y))
            @testset "2nd order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
                quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
                H = inner_product(g_2D, quadrature_interior, quadrature_closure)
                Hi = inverse_inner_product(g_2D, quadrature_interior, quadrature_closure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
            @testset "4th order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
                quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
                H = inner_product(g_2D, quadrature_interior, quadrature_closure)
                Hi = inverse_inner_product(g_2D, quadrature_interior, quadrature_closure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
        end
    end
end
