using Test


using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

using Sbplib.SbpOperators: closure_size, Stencil, VolumeOperator

"""
    monomial(x,k)

Evaluates ``x^k/k!` with the convetion that it is ``0`` for all ``k<0``.
Has the property that ``d/dx monomial(x,k) = monomial(x,k-1)``
"""
function monomial(x,k)
    if k < 0
        return zero(x)
    end
    x^k/factorial(k)
end

@testset "first_derivative" begin
    @testset "Constructors" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)

        g₁ = EquidistantGrid(11, 0., 1.)
        g₂ = EquidistantGrid((11,14), (0.,1.), (1.,3.))
        
        @test first_derivative(g₁, stencil_set, 1) isa TensorMapping{Float64,1,1}
        @test first_derivative(g₂, stencil_set, 2) isa TensorMapping{Float64,2,2}

        interior_stencil = CenteredStencil(-1,0,1)
        closure_stencils = [Stencil(-1,1, center=1)]

        @test first_derivative(g₁, interior_stencil, closure_stencils, 1) isa TensorMapping{Float64,1,1}
        @test first_derivative(g₁, interior_stencil, closure_stencils, 1) isa VolumeOperator
        @test first_derivative(g₂, interior_stencil, closure_stencils, 2) isa TensorMapping{Float64,2,2}
    end

    @testset "Accuracy conditions" begin
        N = 20
        g = EquidistantGrid(N, 0//1,2//1)
        @testset for order ∈ [2,4]
            stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order)
            D₁ = first_derivative(g, stencil_set, 1)

            @testset "boundary x^$k" for k ∈ 0:order÷2
                v = evalOn(g, x->monomial(x,k))

                @testset for i ∈ 1:closure_size(D₁)
                    x, = points(g)[i]
                    @test (D₁*v)[i] == monomial(x,k-1)
                end

                @testset for i ∈ (N-closure_size(D₁)+1):N
                    x, = points(g)[i]
                    @test (D₁*v)[i] == monomial(x,k-1)
                end
            end

            @testset "interior x^$k" for k ∈ 0:order
                v = evalOn(g, x->monomial(x,k))

                x, = points(g)[10]
                @test (D₁*v)[10] == monomial(x,k-1)
            end
        end
    end

    @testset "Accuracy on function" begin
        # 1D
        g = EquidistantGrid(30, 0.,1.)
        v = evalOn(g, x->exp(x))
        @testset for (order, tol) ∈ [(2, 6e-3),(4, 2e-4)]
            stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order)
            D₁ = first_derivative(g, stencil_set, 1)

            @test D₁*v ≈ v rtol=tol
        end

        # 2D
        g = EquidistantGrid((30,60), (0.,0.),(1.,2.))
        v = evalOn(g, (x,y)->exp(0.8x+1.2*y))
        @testset for (order, tol) ∈ [(2, 6e-3),(4, 3e-4)]
            stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order)
            Dx = first_derivative(g, stencil_set, 1)
            Dy = first_derivative(g, stencil_set, 2)

            @test Dx*v ≈ 0.8v rtol=tol
            @test Dy*v ≈ 1.2v rtol=tol
        end
    end
end

