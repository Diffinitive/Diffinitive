using Test

using Diffinitive.Grids
using Diffinitive.RegionIndices
using Diffinitive.LazyTensors

using StaticArrays

@testset "Chart" begin
    X(ξ) = 2ξ
    Grids.jacobian(::typeof(X), ξ) = @SVector[2,2]
    c = Chart(X, unitsquare())
    @test c isa Chart{2}
    @test c([3,2]) == [6,4]
    @test parameterspace(c) == unitsquare()
    @test ndims(c) == 2

    @test jacobian(c, [3,2]) == [2,2]
end

@testset "CartesianAtlas" begin
    c = Chart(identity, unitsquare())

    a = CartesianAtlas([c c; c c])
    @test a isa Atlas
    @test charts(a) == [c c; c c]


    @testset "size" begin
        @test size(CartesianAtlas([c c; c c])) == (2,2)
        @test size(CartesianAtlas([c c c; c c c])) == (2,3)
        @test size(CartesianAtlas([c c; c c; c c])) == (3,2)
    end

    @testset "connections" begin
        a = CartesianAtlas(fill(c, 2,3))
        west = CartesianBoundary{1,LowerBoundary}
        east = CartesianBoundary{1,UpperBoundary}
        south = CartesianBoundary{2,LowerBoundary}
        north = CartesianBoundary{2,UpperBoundary}

        @test Set(connections(a)) == Set([
            (MultiBlockBoundary{(1,1), east}(), MultiBlockBoundary{(2,1), west}()),
            (MultiBlockBoundary{(1,1), north}(), MultiBlockBoundary{(1,2), south}()),
            (MultiBlockBoundary{(2,1), north}(), MultiBlockBoundary{(2,2), south}()),
            (MultiBlockBoundary{(1,2), east}(), MultiBlockBoundary{(2,2), west}()),
            (MultiBlockBoundary{(1,2), north}(), MultiBlockBoundary{(1,3), south}()),
            (MultiBlockBoundary{(2,2), north}(), MultiBlockBoundary{(2,3), south}()),
            (MultiBlockBoundary{(1,3), east}(), MultiBlockBoundary{(2,3), west}()),
        ])
    end
end

@testset "UnstructuredAtlas" begin
    @test_broken false
end
