using Test
using Diffinitive.LazyTensors
using StaticArrays

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

    @testset "size" begin
        v = [@SMatrix[1 3; 2 4] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        cv = ArrayComponentView(v, (1, 1))

        @test size(cv) == (3,4)
    end

    @testset "IndexStyle" begin
        v = [@SMatrix[1 3; 2 4] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        cv = ArrayComponentView(v, (1, 1))

        @test IndexStyle(cv) == IndexStyle(v)
    end

    @testset "_array_type" begin
        v = [@SMatrix[1 3; 2 4] .+ 100*i .+ 10*j for i ∈ 1:3, j∈ 1:4]
        cv = ArrayComponentView(v, (1, 1))

        @test LazyTensors._array_type(cv) == typeof(v)
    end
end
