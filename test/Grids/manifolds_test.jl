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
    @test_broken connections(a) == [
        (
            ((1,1), CartesianBoundary{1,UpperBoundary}()),
            ((1,2), CartesianBoundary{1,LowerBoundary}()),
        ),
        (
            ((1,1), CartesianBoundary{2,LowerBoundary}()),
            ((2,1), CartesianBoundary{2,UpperBoundary}()),
        ),
        (
            ((2,2), CartesianBoundary{1,LowerBoundary}()),
            ((2,1), CartesianBoundary{1,UpperBoundary}()),
        ),
        (
            ((2,2), CartesianBoundary{2,UpperBoundary}()),
            ((1,2), CartesianBoundary{2,LowerBoundary}()),
        )
    ]
end

@testset "UnstructuredAtlas" begin
    @test_broken false
end
