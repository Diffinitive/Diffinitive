using Diffinitive.Grids
using Diffinitive.RegionIndices
using Test
using StaticArrays
using LinearAlgebra


_skew_mapping(a,b) = (ξ̄->ξ̄[1]*a + ξ̄[2]*b, ξ̄->[a  b])

function _partially_curved_mapping()
    x̄((ξ, η)) = @SVector[ξ, η*(1+ξ*(ξ-1))]
    J((ξ, η)) = @SMatrix[
        1         0;
        η*(2ξ-1)  1+ξ*(ξ-1);
    ]

    return x̄, J
end

function _fully_curved_mapping()
    x̄((ξ, η)) = @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
    J((ξ, η)) = @SMatrix[
        2       1-2η;
        (2+η)*ξ 3+1/2*ξ^2;
    ]

    return x̄, J
end

@testset "MappedGrid" begin
    @testset "Constructor" begin
        lg = equidistant_grid((0,0), (1,1), 11, 21)

        x̄ = map(ξ̄ -> 2ξ̄, lg)
        J = map(ξ̄ -> @SArray(fill(2., 2, 2)), lg)
        mg = MappedGrid(lg, x̄, J)

        @test mg isa Grid{SVector{2, Float64},2}
        @test jacobian(mg) isa Array{<:AbstractMatrix}
        @test logical_grid(mg) isa Grid

        @test collect(mg) == x̄
        @test jacobian(mg) == J
        @test logical_grid(mg) == lg


        x̄ = map(ξ̄ -> @SVector[ξ̄[1],ξ̄[2], ξ̄[1] + ξ̄[2]], lg)
        J = map(ξ̄ -> @SMatrix[1 0; 0 1; 1 1], lg)
        mg = MappedGrid(lg, x̄, J)

        @test mg isa Grid{SVector{3, Float64},2}
        @test jacobian(mg) isa Array{<:AbstractMatrix}
        @test logical_grid(mg) isa Grid

        @test collect(mg) == x̄
        @test jacobian(mg) == J
        @test logical_grid(mg) == lg

        sz1 = (10,11)
        sz2 = (10,12)
        @test_throws ArgumentError("Sizes must match") MappedGrid(
            equidistant_grid((0,0), (1,1), sz2...),
            rand(SVector{2},sz1...),
            rand(SMatrix{2,2},sz1...),
        )

        @test_throws ArgumentError("Sizes must match") MappedGrid(
            equidistant_grid((0,0), (1,1), sz1...),
            rand(SVector{2},sz2...),
            rand(SMatrix{2,2},sz1...),
        )

        @test_throws ArgumentError("Sizes must match") MappedGrid(
            equidistant_grid((0,0), (1,1), sz1...),
            rand(SVector{2},sz1...),
            rand(SMatrix{2,2},sz2...),
        )

        err_str = "The size of the jacobian must match the dimensions of the grid and coordinates"
        @test_throws ArgumentError(err_str) MappedGrid(
            equidistant_grid((0,0), (1,1), 10, 11),
            rand(SVector{3}, 10, 11),
            rand(SMatrix{3,4}, 10, 11),
        )

        @test_throws ArgumentError(err_str) MappedGrid(
            equidistant_grid((0,0), (1,1), 10, 11),
            rand(SVector{3}, 10, 11),
            rand(SMatrix{4,2}, 10, 11),
        )
    end

    @testset "Indexing Interface" begin
        lg = equidistant_grid((0,0), (1,1), 11, 21)
        x̄ = map(ξ̄ -> 2ξ̄, lg)
        J = map(ξ̄ -> @SArray(fill(2., 2, 2)), lg)
        mg = MappedGrid(lg, x̄, J)
        @test mg[1,1] == [0.0, 0.0]
        @test mg[4,2] == [0.6, 0.1]
        @test mg[6,10] == [1., 0.9]

        @test mg[begin, begin] == [0.0, 0.0]
        @test mg[end,end] == [2.0, 2.0]
        @test mg[begin,end] == [0., 2.]

        @test axes(mg) == (1:11, 1:21)

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
            @test eachindex(mg) == CartesianIndices((11,21))
        end

        @testset "firstindex" begin
            @test firstindex(mg, 1) == 1
            @test firstindex(mg, 2) == 1
        end

        @testset "lastindex" begin
            @test lastindex(mg, 1) == 11
            @test lastindex(mg, 2) == 21
        end
    end

    @testset "Iterator interface" begin
        lg = equidistant_grid((0,0), (1,1), 11, 21)
        x̄ = map(ξ̄ -> 2ξ̄, lg)
        J = map(ξ̄ -> @SArray(fill(2., 2, 2)), lg)

        mg = MappedGrid(lg, x̄, J)

        lg2 = equidistant_grid((0,0), (1,1), 15, 11)
        sg = MappedGrid(
            equidistant_grid((0,0), (1,1), 15, 11),
            map(ξ̄ -> @SArray[ξ̄[1], ξ̄[2], -ξ̄[1]], lg2), rand(SMatrix{3,2,Float64},15,11)
        )

        @test eltype(mg) == SVector{2,Float64}
        @test eltype(sg) == SVector{3,Float64}

        @test eltype(typeof(mg)) == SVector{2,Float64}
        @test eltype(typeof(sg)) == SVector{3,Float64}

        @test size(mg) == (11,21)
        @test size(sg) == (15,11)

        @test size(mg,2) == 21
        @test size(sg,2) == 11

        @test length(mg) == 231
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
        lg = equidistant_grid((0,0), (1,1), 11, 21)
        x̄ = map(ξ̄ -> 2ξ̄, lg)
        J = map(ξ̄ -> @SArray(fill(2., 2, 2)), lg)
        mg = MappedGrid(lg, x̄, J)

        @test ndims(mg) == 2
    end

    @testset "==" begin
        sz = (15,11)
        lg = equidistant_grid((0,0), (1,1), sz...)
        x = rand(SVector{3,Float64}, sz...)
        J = rand(SMatrix{3,2,Float64}, sz...)

        sg = MappedGrid(lg, x, J)

        sg1 = MappedGrid(equidistant_grid((0,0), (1,1), sz...), copy(x), copy(J))

        sz2 = (15,12)
        lg2 = equidistant_grid((0,0), (1,1), sz2...)
        x2 = rand(SVector{3,Float64}, sz2...)
        J2 = rand(SMatrix{3,2,Float64}, sz2...)
        sg2 = MappedGrid(lg2, x2, J2)

        sg3 = MappedGrid(lg, rand(SVector{3,Float64}, sz...), J)
        sg4 = MappedGrid(lg, x, rand(SMatrix{3,2,Float64}, sz...))

        @test sg == sg1
        @test sg != sg2 # Different size
        @test sg != sg3 # Different coordinates
        @test sg != sg4 # Different jacobian
    end

    @testset "boundary_identifiers" begin
        lg = equidistant_grid((0,0), (1,1), 11, 15)
        x̄ = map(ξ̄ -> 2ξ̄, lg)
        J = map(ξ̄ -> @SArray(fill(2., 2, 2)), lg)
        mg = MappedGrid(lg, x̄, J)
        @test boundary_identifiers(mg) == boundary_identifiers(lg)
    end

    @testset "boundary_indices" begin
        lg = equidistant_grid((0,0), (1,1), 11, 15)
        x̄ = map(ξ̄ -> 2ξ̄, lg)
        J = map(ξ̄ -> @SArray(fill(2., 2, 2)), lg)
        mg = MappedGrid(lg, x̄, J)

        @test boundary_indices(mg, CartesianBoundary{1,LowerBoundary}()) == boundary_indices(lg,CartesianBoundary{1,LowerBoundary}())
        @test boundary_indices(mg, CartesianBoundary{2,LowerBoundary}()) == boundary_indices(lg,CartesianBoundary{2,LowerBoundary}())
        @test boundary_indices(mg, CartesianBoundary{1,UpperBoundary}()) == boundary_indices(lg,CartesianBoundary{1,UpperBoundary}())
    end

    @testset "boundary_grid" begin
        x̄, J = _partially_curved_mapping()
        mg = mapped_grid(x̄, J, 10, 11)
        J1((ξ, η)) = @SMatrix[
            1       ;
            η*(2ξ-1);
        ]
        J2((ξ, η)) = @SMatrix[
            0;
            1+ξ*(ξ-1);
        ]

        function expected_bg(mg, bId, Jb)
            lg = logical_grid(mg)
            return MappedGrid(
                boundary_grid(lg, bId),
                map(x̄, boundary_grid(lg, bId)),
                map(Jb, boundary_grid(lg, bId)),
            )
        end

        let bid = TensorGridBoundary{1, LowerBoundary}()
            @test boundary_grid(mg, bid) == expected_bg(mg, bid, J2)
        end

        let bid = TensorGridBoundary{1, UpperBoundary}()
            @test boundary_grid(mg, bid) == expected_bg(mg, bid, J2)
        end

        let bid = TensorGridBoundary{2, LowerBoundary}()
            @test boundary_grid(mg, bid) == expected_bg(mg, bid, J1)
        end

        let bid = TensorGridBoundary{2, UpperBoundary}()
            @test boundary_grid(mg, bid) == expected_bg(mg, bid, J1)
        end
    end
end

@testset "mapped_grid" begin
    x̄, J = _partially_curved_mapping()
    mg = mapped_grid(x̄, J, 10, 11)
    @test mg isa MappedGrid{SVector{2,Float64}, 2}

    lg = equidistant_grid((0,0), (1,1), 10, 11)
    @test logical_grid(mg) == lg
    @test collect(mg) == map(x̄, lg)

    @test mapped_grid(lg, x̄, J) == mg
end

@testset "jacobian_determinant" begin
    x̄((ξ, η)) = @SVector[ξ*η, ξ + η^2]
    J((ξ, η)) = @SMatrix[
        η    ξ;
        1   2η;
    ]

    g = mapped_grid(x̄, J, 10, 11)
    J = map(logical_grid(g)) do (ξ,η)
        2η^2 - ξ
    end
    @test jacobian_determinant(g) ≈ J


    lg = equidistant_grid((0,0), (1,1), 11, 21)
    x̄ = map(ξ̄ -> @SVector[ξ̄[1],ξ̄[2], ξ̄[1] + ξ̄[2]], lg)
    J = map(ξ̄ -> @SMatrix[1 0; 0 1; 1 1], lg)
    mg = MappedGrid(lg, x̄, J)

    @test_broken jacobian(mg) isa AbstractArray{2,Float64}
end

@testset "metric_tensor" begin
    x̄((ξ, η)) = @SVector[ξ*η, ξ + η^2]
    J((ξ, η)) = @SMatrix[
        η    ξ;
        1   2η;
    ]

    g = mapped_grid(x̄, J, 10, 11)
    G = map(logical_grid(g)) do (ξ,η)
        @SMatrix[
            1+η^2   ξ*η+2η;
            ξ*η+2η  ξ^2 + 4η^2;
        ]
    end
    @test metric_tensor(g) ≈ G
end

@testset "metric_tensor_inverse" begin
    x̄((ξ, η)) = @SVector[ξ + ξ^2/2, η + η^2 + ξ^2/2]
    J((ξ, η)) = @SMatrix[
        1+ξ   0;
        ξ    1+η;
    ]

    g = mapped_grid(x̄, J, 10, 11)
    G⁻¹ = map(logical_grid(g)) do (ξ,η)
        @SMatrix[
            (1+η)^2  -ξ*(1+η);
            -ξ*(1+η) (1+ξ)^2+ξ^2;
        ]/(((1+ξ)^2+ξ^2)*(1+η)^2 - ξ^2*(1+η)^2)

    end

    @test metric_tensor_inverse(g) ≈ G⁻¹
end

@testset "min_spacing" begin
    let g = mapped_grid(identity, x->@SMatrix[1], 11)
        @test min_spacing(g) ≈ 0.1
    end

    let g = mapped_grid(x->x+x.^2/2, x->@SMatrix[1 .+ x], 11)
        @test min_spacing(g) ≈ 0.105
    end

    let g = mapped_grid(x->x + x.*(1 .- x)/2, x->@SMatrix[1.5 .- x], 11)
        @test min_spacing(g) ≈ 0.055
    end

    let g = mapped_grid(identity, x->@SMatrix[1 0; 0 1], 11,11)
        @test min_spacing(g) ≈ 0.1
    end

    let g = mapped_grid(identity, x->@SMatrix[1 0; 0 1], 11,21)
        @test min_spacing(g) ≈ 0.05
    end


    @testset let a = @SVector[1,0], b = @SVector[1,1]/√2
        g = mapped_grid(_skew_mapping(a,b)...,11,11)

        @test min_spacing(g) ≈ 0.1*norm(b-a)
    end

    @testset let a = @SVector[1,0], b = @SVector[-1,1]/√2
        g = mapped_grid(_skew_mapping(a,b)...,11,11)

        @test min_spacing(g) ≈ 0.1*norm(a+b)
    end
end

@testset "normal" begin
    g = mapped_grid(_partially_curved_mapping()...,10, 11)

    @test normal(g, CartesianBoundary{1,LowerBoundary}()) == fill(@SVector[-1,0], 11)
    @test normal(g, CartesianBoundary{1,UpperBoundary}()) == fill(@SVector[1,0], 11)
    @test normal(g, CartesianBoundary{2,LowerBoundary}()) == fill(@SVector[0,-1], 10)
    @test normal(g, CartesianBoundary{2,UpperBoundary}()) ≈ map(boundary_grid(g,CartesianBoundary{2,UpperBoundary}())|>logical_grid) do ξ̄
        α = 1-2ξ̄[1]
        @SVector[α,1]/√(α^2 + 1)
    end

    g = mapped_grid(_fully_curved_mapping()...,5,4)

    unit(v) = v/norm(v)
    @testset let bId = CartesianBoundary{1,LowerBoundary}()
        lbg = boundary_grid(logical_grid(g), bId)
        @test normal(g, bId) ≈ map(lbg) do (ξ, η)
            -unit(@SVector[1/2,  η/3-1/6])
        end
    end

    @testset let bId = CartesianBoundary{1,UpperBoundary}()
        lbg = boundary_grid(logical_grid(g), bId)
        @test normal(g, bId) ≈ map(lbg) do (ξ, η)
            unit(@SVector[7/2, 2η-1]/(5 + 3η + 2η^2))
        end
    end

    @testset let bId = CartesianBoundary{2,LowerBoundary}()
        lbg = boundary_grid(logical_grid(g), bId)
        @test normal(g, bId) ≈ map(lbg) do (ξ, η)
            -unit(@SVector[-2ξ, 2]/(6 + ξ^2 - 2ξ))
        end
    end

    @testset let bId = CartesianBoundary{2,UpperBoundary}()
        lbg = boundary_grid(logical_grid(g), bId)
        @test normal(g, bId) ≈ map(lbg) do (ξ, η)
            unit(@SVector[-3ξ, 2]/(6 + ξ^2 + 3ξ))
        end
    end
end
