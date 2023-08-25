using Sbplib.Grids
using Test
using StaticArrays

@testset "CurvilinearGrid" begin
    g = equidistant_grid((10,10), (0,0), (1,1))
    x̄ = map(ξ̄ -> 2ξ̄, g)
    J = map(ξ̄ -> @SArray(fill(2., 2, 2)), g)

    @test CurvilinearGrid(g, x̄, J) isa Grid{SVector{2, Float64},2}

    cg = CurvilinearGrid(g, x̄, J)
    @test jacobian(cg) isa Array{<:AbstractMatrix}
    @test logicalgrid(cg) isa Grid


    @testset "Indexing Interface" begin
        # cg = CurvilinearGrid(g, x̄, J)
        # @test cg[1,1] == [0.0, 0.0]
        # @test cg[4,2] == [3/9,1/9]
        # @test cg[6,10] == [5/9, 1]

        # @test cg[begin, begin] == [0.0, 0.0]
        # @test cg[end,end] == [1.0, 1.0]
        # @test cg[begin,end] == [0., 1.]

        # @test eachindex(cg) == 1:101
    end

    @testset "Iterator interface" begin
        # @test eltype(EquidistantGrid(0:10)) == Int
        # @test eltype(EquidistantGrid(0:2:10)) == Int
        # @test eltype(EquidistantGrid(0:0.1:10)) == Float64

        # @test size(EquidistantGrid(0:10)) == (11,)
        # @test size(EquidistantGrid(0:0.1:10)) == (101,)

        # @test collect(EquidistantGrid(0:0.1:0.5)) == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        # @test Base.IteratorSize(EquidistantGrid{Float64, StepRange{Float64}}) == Base.HasShape{1}()
    end

    @testset "Base" begin
        # @test ndims(EquidistantGrid(0:10)) == 1
    end

    @testset "boundary_identifiers" begin
        # g = EquidistantGrid(0:0.1:10)
        # @test boundary_identifiers(g) == (Lower(), Upper())
        # @inferred boundary_identifiers(g)
    end

    @testset "boundary_grid" begin
        # g = EquidistantGrid(0:0.1:1)
        # @test boundary_grid(g, Lower()) == ZeroDimGrid(0.0)
        # @test boundary_grid(g, Upper()) == ZeroDimGrid(1.0)
    end

    @testset "refine" begin
        # g = EquidistantGrid(0:0.1:1)
        # @test refine(g, 1) == g
        # @test refine(g, 2) == EquidistantGrid(0:0.05:1)
        # @test refine(g, 3) == EquidistantGrid(0:(0.1/3):1)
    end

    @testset "coarsen" begin
        # g = EquidistantGrid(0:1:10)
        # @test coarsen(g, 1) == g
        # @test coarsen(g, 2) == EquidistantGrid(0:2:10)

        # g = EquidistantGrid(0:0.1:1)
        # @test coarsen(g, 1) == g
        # @test coarsen(g, 2) == EquidistantGrid(0:0.2:1)

        # g = EquidistantGrid(0:10)
        # @test coarsen(g, 1) == EquidistantGrid(0:1:10)
        # @test coarsen(g, 2) == EquidistantGrid(0:2:10)

        # @test_throws DomainError(3, "Size minus 1 must be divisible by the ratio.") coarsen(g, 3)
    end
end
