using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
using Diffinitive.LazyTensors
using Diffinitive.RegionIndices
import Diffinitive.SbpOperators.BoundaryOperator

using StaticArrays
using LinearAlgebra

@testset "normal_derivative" begin
	stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)

    @testset "EquidistantGrid" begin
        g_1D = equidistant_grid(0.0, 1.0, 11)

        d_l = normal_derivative(g_1D, stencil_set, LowerBoundary())
        @test d_l == normal_derivative(g_1D, stencil_set, LowerBoundary())
        @test d_l isa BoundaryOperator{T,LowerBoundary} where T
        @test d_l isa LazyTensor{T,0,1} where T
    end

    @testset "TensorGrid" begin
        g_2D = equidistant_grid((0.0, 0.0), (1.0,1.0), 11, 12)
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

    @testset "MappedGrid" begin
        c = Chart(unitsquare()) do (ξ,η)
            @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
        end
        Grids.jacobian(c::typeof(c), (ξ,η)) = @SMatrix[2 1-2η; (2+η)*ξ 3+ξ^2/2]
        mg = equidistant_grid(c, 10,13)


        # x̄((ξ, η)) = @SVector[ξ, η*(1+ξ*(ξ-1))]
        # J((ξ, η)) = @SMatrix[
        #     1         0;
        #     η*(2ξ-1)  1+ξ*(ξ-1);
        # ]
        # mg = mapped_grid(x̄, J, 20, 21)


        # x̄((ξ, η)) = @SVector[ξ,η]
        # J((ξ, η)) = @SMatrix[
        #     1  0;
        #     0  1;
        # ]
        # mg = mapped_grid(identity, J, 10, 11)

        for bid ∈ boundary_identifiers(mg)
            @testset let bid=bid
                @test normal_derivative(mg, stencil_set, bid) isa LazyTensor{<:Any, 1, 2}
            end
        end

        @testset "Consistency" begin
            v = map(identity, mg)

             @testset "4nd order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)

                for bid ∈ boundary_identifiers(mg)
                    @testset let bid=bid
                        d = normal_derivative(mg, stencil_set, bid)
                        @test d*v ≈ normal(mg, bid) rtol=1e-13
                    end
                end
             end
        end

        @testset "Accuracy" begin
            v = function(x̄)
                sin(norm(x̄+@SVector[1,1]))
            end
            ∇v = function(x̄)
                ȳ = x̄+@SVector[1,1]
                cos(norm(ȳ))*(ȳ/norm(ȳ))
            end

            mg = equidistant_grid(c, 80,80)
            v̄ = map(v, mg)

            @testset for (order, atol) ∈ [(2,4e-2),(4,2e-3)]
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=order)

                @testset for bId ∈ boundary_identifiers(mg)
                    ∂ₙv = map(boundary_grid(mg,bId),normal(mg,bId)) do x̄,n̂
                        n̂⋅∇v(x̄)
                    end

                    dₙ = normal_derivative(mg, stencil_set, bId)
                    @test dₙ*v̄ ≈ ∂ₙv atol=atol
                end
            end
        end
    end
end
