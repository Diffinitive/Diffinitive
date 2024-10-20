using Test
using Diffinitive.Grids
using Diffinitive.LazyTensors
using StaticArrays

@testset "Grid" begin
    struct DummyGrid{T,D} <: Grid{T,D} end

    @test eltype(DummyGrid{Int, 2}) == Int
    @test eltype(DummyGrid{Int, 2}()) == Int

    @test ndims(DummyGrid{Int, 2}()) == 2

    @test coordinate_size(DummyGrid{Int, 1}()) == 1
    @test coordinate_size(DummyGrid{SVector{3,Float64}, 2}()) == 3

    @test coordinate_size(DummyGrid{SVector{3,Float64}, 2}) == 3
end

@testset "component_type" begin
    @test component_type(DummyGrid{Int,1}()) == Int
    @test component_type(DummyGrid{Float64,1}()) == Float64
    @test component_type(DummyGrid{Rational,1}()) == Rational

    @test component_type(DummyGrid{SVector{3,Int},2}()) == Int
    @test component_type(DummyGrid{SVector{2,Float64},3}()) == Float64
    @test component_type(DummyGrid{SVector{4,Rational},4}()) == Rational

    @test component_type(DummyGrid{Float64,1}) == Float64
    @test component_type(DummyGrid{SVector{2,Float64},3}) == Float64

    @test component_type(fill(@SVector[1,2], 4,2)) == Int
end

@testset "eval_on" begin
    @test eval_on(ZeroDimGrid(@SVector[1.,2.]), x̄->x̄[1]+x̄[2]) isa LazyArray
    @test eval_on(ZeroDimGrid(@SVector[1.,2.]), x̄->x̄[1]+x̄[2]) == fill(3.)
    @test eval_on(ZeroDimGrid(@SVector[3.,2.]), x̄->x̄[1]+x̄[2]) == fill(5.)

    @test eval_on(ZeroDimGrid(1.), x̄->2x̄) isa LazyArray
    @test eval_on(ZeroDimGrid(1.), x̄->2x̄) == fill(2.)

    @test eval_on(ZeroDimGrid(@SVector[1.,2.]), π) isa LazyArray
    @test eval_on(ZeroDimGrid(@SVector[1.,2.]), π) == fill(π)

    @test eval_on(EquidistantGrid(range(0,1,length=4)), x->2x) isa LazyArray
    @test eval_on(EquidistantGrid(range(0,1,length=4)), x->2x) == 2 .* range(0,1,length=4)


    g = equidistant_grid((0.0,0.0), (2.0,1.0), 5, 3)

    @test eval_on(g, x̄ -> 0.) isa LazyArray
    @test eval_on(g, x̄ -> 0.) == fill(0., (5,3))

    @test eval_on(g, x̄ -> sin(x̄[1])*cos(x̄[2])) == map(x̄->sin(x̄[1])*cos(x̄[2]), g)

    @test eval_on(g, π) == fill(π, (5,3))

    # Vector valued function
    @test eval_on(g, x̄ -> @SVector[x̄[2], x̄[1]]) isa LazyArray{SVector{2,Float64}}
    @test eval_on(g, x̄ -> @SVector[x̄[2], x̄[1]]) == map(x̄ -> @SVector[x̄[2], x̄[1]], g)

    # Multi-argument functions
    f(x,y) = sin(x)*cos(y)
    @test eval_on(g, f) == map(x̄->f(x̄...), g)
end

@testset "componentview" begin
    v = [@SMatrix[1 3; 2 4] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]

    @test componentview(v, 1, 1) isa AbstractArray
    @test componentview(v, 1, :) isa AbstractArray

    A = @SMatrix[
            1 4 7;
            2 5 8;
            3 6 9;
        ]
    v = [A .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
    @test componentview(v, 2:3, 1:2) isa AbstractArray

    # Correctness of the result is tested in ArrayComponentView
end

@testset "ArrayComponentView" begin
    v = [@SMatrix[1 3; 2 4] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]

    @testset "==" begin
        @test ArrayComponentView(v, (1,1)) == ArrayComponentView(v, (1,1))
        @test ArrayComponentView(v, (1,1)) == ArrayComponentView(copy(v), (1,1))
        @test ArrayComponentView(v, (1,1)) == [1 .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        @test [1 .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4] == ArrayComponentView(v, (1,1))
    end

    @testset "components" begin
        v = [@SMatrix[1 3; 2 4] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]

        @test ArrayComponentView(v, (1, 1))  == [1 .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        @test ArrayComponentView(v, (1, 2))  == [3 .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        @test ArrayComponentView(v, (2, 1))  == [2 .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]

        @test ArrayComponentView(v, (1, :))  == [@SVector[1,3] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        @test ArrayComponentView(v, (2, :))  == [@SVector[2,4] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        @test ArrayComponentView(v, (:, 1))  == [@SVector[1,2] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        @test ArrayComponentView(v, (:, 2))  == [@SVector[3,4] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]


        A = @SMatrix[
            1 4 7;
            2 5 8;
            3 6 9;
        ]
        v = [A .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        @test ArrayComponentView(v, (1:2, 1:2)) == [@SMatrix[1 4;2 5] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        @test ArrayComponentView(v, (2:3, 1:2)) == [@SMatrix[2 5;3 6] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
    end
end

@testset "_ncomponents" begin
    @test Grids._ncomponents(Int) == 1
    @test Grids._ncomponents(Float64) == 1
    @test Grids._ncomponents(Rational) == 1

    @test Grids._ncomponents(SVector{3,Int}) == 3
    @test Grids._ncomponents(SVector{2,Float64}) == 2
    @test Grids._ncomponents(SVector{4,Rational}) == 4
end
