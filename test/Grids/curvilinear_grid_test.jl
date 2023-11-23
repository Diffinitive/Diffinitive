using Sbplib.Grids
using Test
using StaticArrays

@testset "CurvilinearGrid" begin
    lg = equidistant_grid((11,11), (0,0), (1,1)) # TODO: Change dims of the grid to be different
    x̄ = map(ξ̄ -> 2ξ̄, lg)
    J = map(ξ̄ -> @SArray(fill(2., 2, 2)), lg)
    cg = CurvilinearGrid(lg, x̄, J)

    # TODO: Test constructor for different dims of range and domain for the coordinates
    # TODO: Test constructor with different type than TensorGrid. a dummy type?

    @test_broken false # @test_throws ArgumentError("Sizes must match") CurvilinearGrid(lg, map(ξ̄ -> @SArray[ξ̄[1], ξ̄[2], -ξ̄[1]], lg), rand(SMatrix{2,3,Float64},15,11))


    @test cg isa Grid{SVector{2, Float64},2}

    @test jacobian(cg) isa Array{<:AbstractMatrix}
    @test logicalgrid(cg) isa Grid

    @testset "Indexing Interface" begin
        cg = CurvilinearGrid(lg, x̄, J)
        @test cg[1,1] == [0.0, 0.0]
        @test cg[4,2] == [0.6, 0.2]
        @test cg[6,10] == [1., 1.8]

        @test cg[begin, begin] == [0.0, 0.0]
        @test cg[end,end] == [2.0, 2.0]
        @test cg[begin,end] == [0., 2.]

        @test eachindex(cg) == CartesianIndices((11,11))

        @testset "cartesian indexing" begin
            cases = [
                 (1,1) ,
                 (3,5) ,
                 (10,6),
                 (1,1) ,
                 (3,2) ,
            ]

            @testset "i = $is" for (lg, is) ∈ cases
                @test cg[CartesianIndex(is...)] == cg[is...]
            end
        end

        @testset "eachindex" begin
            @test eachindex(cg) == CartesianIndices((11,11))
        end

        @testset "firstindex" begin
            @test firstindex(cg, 1) == 1
            @test firstindex(cg, 2) == 1
        end

        @testset "lastindex" begin
            @test lastindex(cg, 1) == 11
            @test lastindex(cg, 2) == 11
        end
    end
    # TODO: Test with different types of logical grids

    @testset "Iterator interface" begin
        sg = CurvilinearGrid(
            equidistant_grid((15,11), (0,0), (1,1)),
            map(ξ̄ -> @SArray[ξ̄[1], ξ̄[2], -ξ̄[1]], lg), rand(SMatrix{2,3,Float64},15,11)
        )

        @test eltype(cg) == SVector{2,Float64}
        @test eltype(sg) == SVector{3,Float64}

        @test eltype(typeof(cg)) == SVector{2,Float64}
        @test eltype(typeof(sg)) == SVector{3,Float64}

        @test size(cg) == (11,11)
        @test size(sg) == (15,11)

        @test size(cg,2) == 11
        @test size(sg,2) == 11

        @test length(cg) == 121
        @test length(sg) == 165

        @test Base.IteratorSize(cg) == Base.HasShape{2}()
        @test Base.IteratorSize(typeof(cg)) == Base.HasShape{2}()

        @test Base.IteratorSize(sg) == Base.HasShape{2}()
        @test Base.IteratorSize(typeof(sg)) == Base.HasShape{2}()

        element, state = iterate(cg)
        @test element == lg[1,1].*2
        element, _ =  iterate(cg, state)
        @test element == lg[2,1].*2

        element, state = iterate(sg)
        @test element == sg.physicalcoordinates[1,1]
        element, _ = iterate(sg, state)
        @test element == sg.physicalcoordinates[2,1]

        @test collect(cg) == 2 .* lg
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
