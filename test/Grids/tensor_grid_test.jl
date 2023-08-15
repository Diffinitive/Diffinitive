using Test
using Sbplib.Grids
using StaticArrays
using Sbplib.RegionIndices

@testset "TensorGrid" begin
    g₁ = EquidistantGrid(range(0,1,length=11))
    g₂ = EquidistantGrid(range(2,3,length=6))
    g₃ = EquidistantGrid(1:10)
    g₄ = ZeroDimGrid(@SVector[1,2])

    @test TensorGrid(g₁, g₂) isa TensorGrid
    @test TensorGrid(g₁, g₂) isa Grid{SVector{2,Float64}, 2}
    @test TensorGrid(g₃, g₃) isa Grid{SVector{2,Int}, 2}
    @test TensorGrid(g₁, g₂, g₃) isa Grid{SVector{3,Float64}, 3}
    @test TensorGrid(g₁, g₄) isa Grid{SVector{3,Float64}, 1}
    @test TensorGrid(g₁, g₄, g₂) isa Grid{SVector{4,Float64}, 2}

    @testset "Indexing Interface" begin
        @testset "regular indexing" begin
            @test TensorGrid(g₁, g₂)[1,1] isa SVector{2,Float64}
            @test TensorGrid(g₁, g₂)[1,1] == [0.0,2.0]
            @test TensorGrid(g₁, g₂)[3,5] == [0.2,2.8]
            @test TensorGrid(g₁, g₂)[10,6] == [0.9,3.0]

            @test TensorGrid(g₁, g₃)[1,1] isa SVector{2,Float64}
            @test TensorGrid(g₁, g₃)[1,1] == [0.0,1.0]

            @test TensorGrid(g₁, g₂, g₃)[3,4,5] isa SVector{3,Float64}
            @test TensorGrid(g₁, g₂, g₃)[3,4,5] == [0.2, 2.6, 5.0]

            @test TensorGrid(g₁, g₄)[3] isa SVector{3,Float64}
            @test TensorGrid(g₁, g₄)[3] == [0.2, 1., 2.]

            @test TensorGrid(g₁, g₄, g₂)[3,2] isa SVector{4,Float64}
            @test TensorGrid(g₁, g₄, g₂)[3,2] == [0.2, 1., 2., 2.2]
        end

        @testset "cartesian indexing" begin
            cases = [
                (TensorGrid(g₁, g₂),     (1,1)  ),
                (TensorGrid(g₁, g₂),     (3,5)  ),
                (TensorGrid(g₁, g₂),     (10,6) ),
                (TensorGrid(g₁, g₃),     (1,1)  ),
                (TensorGrid(g₁, g₂, g₃), (3,4,5)),
                (TensorGrid(g₁, g₄),     (3)    ),
                (TensorGrid(g₁, g₄, g₂), (3,2)  ),
            ]

            @testset "i = $is" for (g, is) ∈ cases
                @test g[CartesianIndex(is...)] == g[is...]
            end
        end

        @testset "eachindex" begin
            @test eachindex(TensorGrid(g₁, g₂)) == CartesianIndices((11,6))
            @test eachindex(TensorGrid(g₁, g₃)) == CartesianIndices((11,10))
            @test eachindex(TensorGrid(g₁, g₂, g₃)) == CartesianIndices((11,6,10))
            @test eachindex(TensorGrid(g₁, g₄)) == CartesianIndices((11,))
            @test eachindex(TensorGrid(g₁, g₄, g₂)) == CartesianIndices((11,6))
        end

        @testset "firstindex" begin
            @test_broken firstindex(TensorGrid(g₁, g₂, g₃), 1) == 1
            @test_broken firstindex(TensorGrid(g₁, g₂, g₃), 2) == 1
            @test_broken firstindex(TensorGrid(g₁, g₂, g₃), 3) == 1
        end

        @testset "lastindex" begin
            @test_broken lastindex(TensorGrid(g₁, g₂, g₃), 1) == 11
            @test_broken lastindex(TensorGrid(g₁, g₂, g₃), 2) == 6
            @test_broken lastindex(TensorGrid(g₁, g₂, g₃), 3) == 10
        end
    end

    @testset "Iterator interface" begin
        @test eltype(TensorGrid(g₁, g₂)) == SVector{2,Float64}
        @test eltype(TensorGrid(g₁, g₃)) == SVector{2,Float64}
        @test eltype(TensorGrid(g₁, g₂, g₃)) == SVector{3,Float64}
        @test eltype(TensorGrid(g₁, g₄)) == SVector{3,Float64}
        @test eltype(TensorGrid(g₁, g₄, g₂)) == SVector{4,Float64}

        @test size(TensorGrid(g₁, g₂)) == (11,6)
        @test size(TensorGrid(g₁, g₃)) == (11,10)
        @test size(TensorGrid(g₁, g₂, g₃)) == (11,6,10)
        @test size(TensorGrid(g₁, g₄)) == (11,)
        @test size(TensorGrid(g₁, g₄, g₂)) == (11,6)

        @test Base.IteratorSize(TensorGrid(g₁, g₂)) == Base.HasShape{2}()
        @test Base.IteratorSize(TensorGrid(g₁, g₃)) == Base.HasShape{2}()
        @test Base.IteratorSize(TensorGrid(g₁, g₂, g₃)) == Base.HasShape{3}()
        @test Base.IteratorSize(TensorGrid(g₁, g₄)) == Base.HasShape{1}()
        @test Base.IteratorSize(TensorGrid(g₁, g₄, g₂)) == Base.HasShape{2}()

        @test iterate(TensorGrid(g₁, g₂))[1] isa SVector{2,Float64}
        @test iterate(TensorGrid(g₁, g₃))[1] isa SVector{2,Float64}
        @test iterate(TensorGrid(g₁, g₂, g₃))[1] isa SVector{3,Float64}
        @test iterate(TensorGrid(g₁, g₄))[1] isa SVector{3,Float64}
        @test iterate(TensorGrid(g₁, g₄, g₂))[1] isa SVector{4,Float64}

        @test collect(TensorGrid(g₁, g₂)) == [@SVector[x,y] for x ∈ range(0,1,length=11), y ∈ range(2,3,length=6)]
        @test collect(TensorGrid(g₁, g₃)) == [@SVector[x,y] for x ∈ range(0,1,length=11), y ∈ 1:10]
        @test collect(TensorGrid(g₁, g₂, g₃)) == [@SVector[x,y,z] for x ∈ range(0,1,length=11), y ∈ range(2,3,length=6), z ∈ 1:10]
        @test collect(TensorGrid(g₁, g₄)) == [@SVector[x,1,2] for x ∈ range(0,1,length=11)]
        @test collect(TensorGrid(g₁, g₄, g₂)) == [@SVector[x,1,2,y] for x ∈ range(0,1,length=11), y ∈ range(2,3,length=6)]
    end

    @testset "refine" begin
        g1(n) = EquidistantGrid(range(0,1,length=n))
        g2(n) = EquidistantGrid(range(2,3,length=n))

        @test refine(TensorGrid(g1(11), g2(6)),1) == TensorGrid(g1(11), g2(6))
        @test refine(TensorGrid(g1(11), g2(6)),2) == TensorGrid(g1(21), g2(11))
        @test refine(TensorGrid(g1(11), g2(6)),3) == TensorGrid(g1(31), g2(16))
        @test refine(TensorGrid(g1(11), g₄), 1) == TensorGrid(g1(11), g₄)
        @test refine(TensorGrid(g1(11), g₄), 2) == TensorGrid(g1(21), g₄)
    end

    @testset "coarsen" begin
        g1(n) = EquidistantGrid(range(0,1,length=n))
        g2(n) = EquidistantGrid(range(2,3,length=n))

        @test coarsen(TensorGrid(g1(11), g2(6)),1) == TensorGrid(g1(11), g2(6))
        @test coarsen(TensorGrid(g1(21), g2(11)),2) == TensorGrid(g1(11), g2(6))
        @test coarsen(TensorGrid(g1(31), g2(16)),3) == TensorGrid(g1(11), g2(6))
        @test coarsen(TensorGrid(g1(11), g₄), 1) == TensorGrid(g1(11), g₄)
        @test coarsen(TensorGrid(g1(21), g₄), 2) == TensorGrid(g1(11), g₄)
    end

    @testset "boundary_identifiers" begin
        @test boundary_identifiers(TensorGrid(g₁, g₂)) == map((n,id)->TensorGridBoundary{n,id}(), (1,1,2,2), (Lower,Upper,Lower,Upper))
        @test boundary_identifiers(TensorGrid(g₁, g₄)) == (TensorGridBoundary{1,Lower}(),TensorGridBoundary{1,Upper}())
    end

    @testset "boundary_grid" begin
        @test boundary_grid(TensorGrid(g₁, g₂), TensorGridBoundary{1, Upper}()) == TensorGrid(ZeroDimGrid(g₁[end]), g₂)
        @test boundary_grid(TensorGrid(g₁, g₄), TensorGridBoundary{1, Upper}()) == TensorGrid(ZeroDimGrid(g₁[end]), g₄)
    end
end

@testset "combined_coordinate_vector_type" begin
    @test Grids.combined_coordinate_vector_type(Float64) == Float64
    @test Grids.combined_coordinate_vector_type(Float64, Int) == SVector{2,Float64}
    @test Grids.combined_coordinate_vector_type(Float32, Int16, Int32) == SVector{3,Float32}

    @test Grids.combined_coordinate_vector_type(SVector{2,Float64}) == SVector{2,Float64}
    @test Grids.combined_coordinate_vector_type(SVector{2,Float64}, SVector{1,Float64}) == SVector{3,Float64}
    @test Grids.combined_coordinate_vector_type(SVector{2,Float64}, SVector{1,Int}, SVector{3, Float32}) == SVector{6,Float64}
end

@testset "combine_coordinates" begin
    @test Grids.combine_coordinates(1,2,3) isa SVector{3, Int}
    @test Grids.combine_coordinates(1,2,3) == [1,2,3]
    @test Grids.combine_coordinates(1,2.,3) isa SVector{3, Float64}
    @test Grids.combine_coordinates(1,2.,3) == [1,2,3]
    @test Grids.combine_coordinates(1,@SVector[2.,3]) isa SVector{3, Float64}
    @test Grids.combine_coordinates(1,@SVector[2.,3]) == [1,2,3]
end
