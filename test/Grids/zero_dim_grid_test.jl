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
        @test refine(ZeroDimGrid(@SVector[1.0,2.0]),1) == ZeroDimGrid(@SVector[1.0,2.0])
        @test refine(ZeroDimGrid(@SVector[1.0,2.0]),2) == ZeroDimGrid(@SVector[1.0,2.0])
    end

    @testset "coarsen" begin
        @test coarsen(ZeroDimGrid(@SVector[1.0,2.0]),1) == ZeroDimGrid(@SVector[1.0,2.0])
        @test coarsen(ZeroDimGrid(@SVector[1.0,2.0]),2) == ZeroDimGrid(@SVector[1.0,2.0])
    end

    @testset "boundary_identifiers" begin
        @test boundary_identifiers(ZeroDimGrid(@SVector[1.0,2.0])) == ()
    end

    @testset "boundary_grid" begin
        @test_throws ArgumentError("ZeroDimGrid has no boundaries") boundary_grid(ZeroDimGrid(@SVector[1.0,2.0]), :bid)
    end

    @testset "boundary_indices" begin
        @test_throws ArgumentError("ZeroDimGrid has no boundaries") boundary_indices(ZeroDimGrid(@SVector[1.0,2.0]), :bid)
    end
end
