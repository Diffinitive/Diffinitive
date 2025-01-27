using Test

using Diffinitive.Grids
using Diffinitive.RegionIndices
using Diffinitive.LazyTensors

# using StaticArrays

@testset "ParameterSpace" begin
    @test ndims(HyperBox([1,1], [2,2])) == 2
    @test ndims(unittetrahedron()) == 3
end

@testset "Interval" begin
    @test Interval <: ParameterSpace{1}

    @test Interval(0,1) isa Interval{Int}
    @test Interval(0,1.) isa Interval{Float64}

    @test unitinterval() isa Interval{Float64}
    @test unitinterval() == Interval(0.,1.)
    @test limits(unitinterval()) == (0.,1.)

    @test unitinterval(Int) isa Interval{Int}
    @test unitinterval(Int) == Interval(0,1)
    @test limits(unitinterval(Int)) == (0,1)
end

@testset "HyperBox" begin
    @test HyperBox{<:Any, 2} <: ParameterSpace{2}
    @test HyperBox([1,1], [2,2]) isa HyperBox{Int, 2}

    @test HyperBox([1,2], [1.,2.]) isa HyperBox{Float64,2}

    @test limits(HyperBox([1,2], [3,4])) == ([1,2], [3,4])
    @test limits(HyperBox([1,2], [3,4]), 1) == (1,3)
    @test limits(HyperBox([1,2], [3,4]), 2) == (2,4)

    @test unitsquare() isa HyperBox{Float64,2}
    @test limits(unitsquare()) == ([0,0],[1,1])

    @test unitcube() isa HyperBox{Float64,3}
    @test limits(unitcube()) == ([0,0,0],[1,1,1])

    @test unithyperbox(4) isa HyperBox{Float64,4}
    @test limits(unithyperbox(4)) == ([0,0,0,0],[1,1,1,1])
end

@testset "Simplex" begin
    @test Simplex{<:Any, 3} <: ParameterSpace{3}
    @test Simplex([1,2], [3,4]) isa Simplex{Int, 2}
    @test Simplex([1,2,3], [4,5,6],[1,1,1]) isa Simplex{Int, 3}

    @test Simplex([1,2], [3.,4.]) isa Simplex{Float64, 2}

    @test verticies(Simplex([1,2], [3,4])) == ([1,2], [3,4])

    @test unittriangle() isa Simplex{Float64,2}
    @test verticies(unittriangle()) == ([0,0], [1,0], [0,1])

    @test unittetrahedron() isa  Simplex{Float64,3}
    @test verticies(unittetrahedron()) == ([0,0,0], [1,0,0], [0,1,0],[0,0,1])

    @test unitsimplex(4) isa Simplex{Float64,4}
end

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

@testset "Line" begin
    @test_broken false
end

@testset "LineSegment" begin
    @test_broken false
end

@testset "linesegments" begin
    @test_broken false
end

@testset "polygon_edges" begin
    @test_broken false
end

@testset "Circle" begin
    @test_broken false
end

@testset "TransfiniteInterpolationSurface" begin
    @test_broken false
end
