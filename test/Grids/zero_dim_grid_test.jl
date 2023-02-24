using Test
using Sbplib.Grids
using StaticArrays

@testset "ZeroDimGrid" begin
    @test ZeroDimGrid(1) isa ZeroDimGrid{Int}
    @test ZeroDimGrid([1,2,3]) isa ZeroDimGrid{Vector{Int}}
    @test ZeroDimGrid(@SVector[1.0,2.0]) isa ZeroDimGrid{SVector{2,Float64}}

    @testset "Indexing Interface" begin
        g = ZeroDimGrid(@SVector[1,2])

        @test g[] == [1,2]
        @test eachindex(g) == CartesianIndices(())
    end

    @testset "Iterator interface" begin
        g = ZeroDimGrid(@SVector[1,2])

        @test Base.IteratorSize(g) == Base.HasShape{0}()
        @test eltype(g) == SVector{2,Int}
        @test length(g) == 1
        @test size(g) == ()
        @test collect(g) == fill(@SVector[1,2])
    end

    @testset "refine" begin
        @test_broken false
    end

    @testset "coarsen" begin
        @test_broken false
    end

    @testset "boundary_identifiers" begin
        @test_broken false
    end

    @testset "boundary_grid" begin
        @test_broken false
        # Test that it throws an error
    end
end
