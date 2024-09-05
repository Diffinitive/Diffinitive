using Test

using Sbplib
using Sbplib.Grids
using Sbplib.SbpOperators
using Sbplib.RegionIndices

using SparseArrayKit
using Tokens
using Tullio


@testset "SparseArray" begin
    g = equidistant_grid((0,0),(1,2), 20,30)
    stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)


    @testset let Δ = laplace(g, stencil_set), M = SparseArray(Δ)
        @test ndims(M) == 4
        @test size(M) == (20,30,20,30)

        v = rand(size(g)...)
        @tullio Mv[i,j] := M[i,j,k,l]*v[k,l]

        @test Mv ≈ Δ*v
    end

    @testset let dₙ = normal_derivative(g, stencil_set,CartesianBoundary{1,LowerBoundary}()), M = SparseArray(dₙ)
        @test ndims(M) == 3
        @test size(M) == (30,20,30)

        v = rand(size(g)...)
        @tullio Mv[i] := M[i,j,k]*v[j,k]
        @test Mv ≈ dₙ*v
    end
end
