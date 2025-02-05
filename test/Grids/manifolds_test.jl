using Test

using Diffinitive.Grids
using Diffinitive.RegionIndices
using Diffinitive.LazyTensors

@testset "Chart" begin
    c = Chart(x->2x, unitsquare())
    @test c isa Chart{2}
    @test c([3,2]) == [6,4]
    @test parameterspace(c) == unitsquare()
    @test ndims(c) == 2

    @test_broken jacobian(c, [3,2])
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

    west = CartesianBoundary{1,LowerBoundary}
    east = CartesianBoundary{1,UpperBoundary}
    south = CartesianBoundary{2,LowerBoundary}
    north = CartesianBoundary{2,UpperBoundary}

    @test_broken Set(connections(a)) == Set([
        (MultiBlockBoundary{(1,1), east}, MultiBlockBoundary{(2,1), west}),
        (MultiBlockBoundary{(1,2), east}, MultiBlockBoundary{(2,2), west}),
        (MultiBlockBoundary{(1,1), north}, MultiBlockBoundary{(1,2), south}),
        (MultiBlockBoundary{(1,2), north}, MultiBlockBoundary{(2,2), south}),
    ])
end

@testset "UnstructuredAtlas" begin
    @test_broken false
end
