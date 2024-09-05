using Test

using Sbplib
using Sbplib.Grids
using Sbplib.SbpOperators
using Sbplib.RegionIndices

using SparseArrays
using Tokens


@testset "SparseArray" begin
    g = equidistant_grid((0,0),(1,2), 20,30)
    stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)


    @testset let Δ = laplace(g, stencil_set), M = sparse(Δ)
        @test ndims(M) == 2
        @test size(M) == (20*30,20*30)

        v = rand(size(g)...)

        Mv = M*reshape(v,:)
        @test Mv ≈ reshape(Δ*v,:)
    end

    @testset let dₙ = normal_derivative(g, stencil_set,CartesianBoundary{1,LowerBoundary}()), M = sparse(dₙ)
        @test ndims(M) == 2
        @test size(M) == (30,20*30)

        v = rand(size(g)...)
        Mv = M*reshape(v,:)
        @test Mv ≈ reshape(dₙ*v,:)
    end
end
