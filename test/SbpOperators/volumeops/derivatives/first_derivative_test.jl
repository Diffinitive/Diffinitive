using Test

using Sbplib.SbpOperators
using Sbplib.Grids

using Sbplib.SbpOperators: closure_size

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
    @testset "accuracy" begin
        N = 20
        g = EquidistantGrid(N, 0//1,2//1)
        @testset for order ∈ [2,4]
            stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order)
            D₁ = first_derivative(g, stencil_set, 1)

            @testset "boundary accuracy $k" for k ∈ 0:order÷2
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

            @testset "interior accuracy $k" for k ∈ 0:order
                v = evalOn(g, x->monomial(x,k))

                x, = points(g)[10]
                @test (D₁*v)[10] == monomial(x,k-1)
            end
        end
    end
end

