using Test

using Diffinitive.Grids
using StaticArrays

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

    @test boundary_identifiers(unitinterval()) == (LowerBoundary(), UpperBoundary())

    @test 0 ∈ Interval(0,1)
    @test 0. ∈ Interval(0,1)
    @test 1. ∈ Interval(0,1)
    @test 2 ∉ Interval(0,1)
    @test -1 ∉ Interval(0,1)
    @test -1. ∉ Interval(0,1)
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


    @test boundary_identifiers(unitsquare()) == [
        CartesianBoundary{1,LowerBoundary}(),
        CartesianBoundary{1,UpperBoundary}(),
        CartesianBoundary{2,LowerBoundary}(),
        CartesianBoundary{2,UpperBoundary}(),
    ]

    @test boundary_identifiers(unitcube()) == [
        CartesianBoundary{1,LowerBoundary}(),
        CartesianBoundary{1,UpperBoundary}(),
        CartesianBoundary{2,LowerBoundary}(),
        CartesianBoundary{2,UpperBoundary}(),
        CartesianBoundary{3,LowerBoundary}(),
        CartesianBoundary{3,UpperBoundary}(),
    ]

    @test @SVector[1.5, 3.5] ∈ HyperBox([1,2], [3,4])
    @test @SVector[1, 2] ∈ HyperBox([1,2], [3,4])
    @test @SVector[3, 4] ∈ HyperBox([1,2], [3,4])

    @test @SVector[0.5, 3.5] ∉ HyperBox([1,2], [3,4])
    @test @SVector[1.5, 4.5] ∉ HyperBox([1,2], [3,4])
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

    @testset "Base.in" begin
        @testset "2D" begin
            T₂ = Simplex([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])
            @test [0.1, 0.1] ∈ T₂
            @test [0.3, 0.3] ∈ T₂
            @test [1.0, 0.0] ∈ T₂
            @test [0.0, 0.0] ∈ T₂
            @test [0.0, 1.0] ∈ T₂
            @test [0.5, 0.5] ∈ T₂

            @test [0.6, 0.6] ∉ T₂
            @test [-0.1, 0.1] ∉ T₂
        end

        @testset "3D" begin
            T₃ = Simplex([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
            @test [0.1, 0.1, 0.1] ∈ T₃
            @test [0.0, 0.0, 0.0] ∈ T₃
            @test [1.0, 0.0, 0.0] ∈ T₃
            @test [0.25, 0.25, 0.25] ∈ T₃
            @test [0.5, 0.5, 0.0] ∈ T₃
            @test [0.3, 0.3, 0.3] ∈ T₃

            @test [0.5, 0.5, 1.0] ∉ T₃
            @test [0.3, 0.3, 0.5] ∉ T₃
        end
    end
end
