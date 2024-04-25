using Sbplib.Grids
using Sbplib.RegionIndices
using Test
using StaticArrays

@testset "MappedGrid" begin
    lg = equidistant_grid((11,11), (0,0), (1,1)) # TODO: Change dims of the grid to be different
    x̄ = map(ξ̄ -> 2ξ̄, lg)
    J = map(ξ̄ -> @SArray(fill(2., 2, 2)), lg)
    mg = MappedGrid(lg, x̄, J)

    # TODO: Test constructor for different dims of range and domain for the coordinates
    # TODO: Test constructor with different type than TensorGrid. a dummy type?

    @test_broken false # @test_throws ArgumentError("Sizes must match") MappedGrid(lg, map(ξ̄ -> @SArray[ξ̄[1], ξ̄[2], -ξ̄[1]], lg), rand(SMatrix{2,3,Float64},15,11))


    @test mg isa Grid{SVector{2, Float64},2}

    @test jacobian(mg) isa Array{<:AbstractMatrix}
    @test logicalgrid(mg) isa Grid

    @testset "Indexing Interface" begin
        mg = MappedGrid(lg, x̄, J)
        @test mg[1,1] == [0.0, 0.0]
        @test mg[4,2] == [0.6, 0.2]
        @test mg[6,10] == [1., 1.8]

        @test mg[begin, begin] == [0.0, 0.0]
        @test mg[end,end] == [2.0, 2.0]
        @test mg[begin,end] == [0., 2.]

        @test eachindex(mg) == CartesianIndices((11,11))

        @testset "cartesian indexing" begin
            cases = [
                 (1,1) ,
                 (3,5) ,
                 (10,6),
                 (1,1) ,
                 (3,2) ,
            ]

            @testset "i = $is" for (lg, is) ∈ cases
                @test mg[CartesianIndex(is...)] == mg[is...]
            end
        end

        @testset "eachindex" begin
            @test eachindex(mg) == CartesianIndices((11,11))
        end

        @testset "firstindex" begin
            @test firstindex(mg, 1) == 1
            @test firstindex(mg, 2) == 1
        end

        @testset "lastindex" begin
            @test lastindex(mg, 1) == 11
            @test lastindex(mg, 2) == 11
        end
    end
    # TODO: Test with different types of logical grids

    @testset "Iterator interface" begin
        sg = MappedGrid(
            equidistant_grid((15,11), (0,0), (1,1)),
            map(ξ̄ -> @SArray[ξ̄[1], ξ̄[2], -ξ̄[1]], lg), rand(SMatrix{2,3,Float64},15,11)
        )

        @test eltype(mg) == SVector{2,Float64}
        @test eltype(sg) == SVector{3,Float64}

        @test eltype(typeof(mg)) == SVector{2,Float64}
        @test eltype(typeof(sg)) == SVector{3,Float64}

        @test size(mg) == (11,11)
        @test size(sg) == (15,11)

        @test size(mg,2) == 11
        @test size(sg,2) == 11

        @test length(mg) == 121
        @test length(sg) == 165

        @test Base.IteratorSize(mg) == Base.HasShape{2}()
        @test Base.IteratorSize(typeof(mg)) == Base.HasShape{2}()

        @test Base.IteratorSize(sg) == Base.HasShape{2}()
        @test Base.IteratorSize(typeof(sg)) == Base.HasShape{2}()

        element, state = iterate(mg)
        @test element == lg[1,1].*2
        element, _ =  iterate(mg, state)
        @test element == lg[2,1].*2

        element, state = iterate(sg)
        @test element == sg.physicalcoordinates[1,1]
        element, _ = iterate(sg, state)
        @test element == sg.physicalcoordinates[2,1]

        @test collect(mg) == 2 .* lg
    end

    @testset "Base" begin
        @test ndims(mg) == 2
    end

    @testset "boundary_identifiers" begin
        @test boundary_identifiers(mg) == boundary_identifiers(lg)
    end

    @testset "boundary_indices" begin
        @test boundary_indices(mg, CartesianBoundary{1,Lower}()) == boundary_indices(lg,CartesianBoundary{1,Lower}())
        @test boundary_indices(mg, CartesianBoundary{2,Lower}()) == boundary_indices(lg,CartesianBoundary{2,Lower}())
        @test boundary_indices(mg, CartesianBoundary{1,Upper}()) == boundary_indices(lg,CartesianBoundary{1,Upper}())
    end

    @testset "boundary_grid" begin
        x̄((ξ, η)) = @SVector[ξ, η*(1+ξ*(ξ-1))]
        J((ξ, η)) = @SMatrix[
            1         0;
            η*(2ξ-1)  1+ξ*(ξ-1);
        ]

        mg = mapped_grid(x̄, J, 10, 11)
        J1((ξ, η)) = @SMatrix[
            1       ;
            η*(2ξ-1);
        ]
        J2((ξ, η)) = @SMatrix[
            0;
            1+ξ*(ξ-1);
        ]

        function test_boundary_grid(mg, bId, Jb)
            bg = boundary_grid(mg, bId)

            lg = logicalgrid(mg)
            expected_bg = MappedGrid(
                boundary_grid(lg, bId),
                map(x̄, boundary_grid(lg, bId)),
                map(Jb, boundary_grid(lg, bId)),
            )

            @testset let bId=bId, bg=bg, expected_bg=expected_bg
                @test collect(bg) == collect(expected_bg)
                @test logicalgrid(bg) == logicalgrid(expected_bg)
                @test jacobian(bg) == jacobian(expected_bg)
                # TODO: Implement equality of a curvilinear grid and simlify the above
            end
        end

        @testset test_boundary_grid(mg, TensorGridBoundary{1, Lower}(), J2)
        @testset test_boundary_grid(mg, TensorGridBoundary{1, Upper}(), J2)
        @testset test_boundary_grid(mg, TensorGridBoundary{2, Lower}(), J1)
        @testset test_boundary_grid(mg, TensorGridBoundary{2, Upper}(), J1)
    end

    @testset "jacobian_determinant" begin
        @test_broken false
    end

    @testset "geometric_tensor" begin
        @test_broken false
    end

    @testset "geometric_tensor_inverse" begin
        @test_broken false
    end

end

@testset "mapped_grid" begin
    x̄((ξ, η)) = @SVector[ξ, η*(1+ξ*(ξ-1))]
    J((ξ, η)) = @SMatrix[
        1         0;
        η*(2ξ-1)  1+ξ*(ξ-1);
    ]
    mg = mapped_grid(x̄, J, 10, 11)
    @test mg isa MappedGrid{SVector{2,Float64}, 2}

    lg = equidistant_grid((10,11), (0,0), (1,1))
    @test logicalgrid(mg) == lg
    @test collect(mg) == map(x̄, lg)
end
