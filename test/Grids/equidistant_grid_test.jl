using Diffinitive.Grids
using Test
using Diffinitive.LazyTensors
using StaticArrays


@testset "EquidistantGrid" begin
    @test EquidistantGrid(0:0.1:10) isa EquidistantGrid
    @test EquidistantGrid(range(0,1,length=10)) isa EquidistantGrid
    @test EquidistantGrid(LinRange(0,1,11)) isa EquidistantGrid

    @testset "Indexing Interface" begin
        g = EquidistantGrid(0:0.1:10)
        @test g[1] == 0.0
        @test g[5] == 0.4
        @test g[101] == 10.0

        @test g[begin] == 0.0
        @test g[end] == 10.0

        @test all(eachindex(g) .== 1:101)

        @test firstindex(g) == 1
        @test lastindex(g) == 101
    end

    @testset "Iterator interface" begin
        @test eltype(EquidistantGrid(0:10)) == Int
        @test eltype(EquidistantGrid(0:2:10)) == Int
        @test eltype(EquidistantGrid(0:0.1:10)) == Float64
        @test size(EquidistantGrid(0:10)) == (11,)
        @test size(EquidistantGrid(0:0.1:10)) == (101,)

        @test size(EquidistantGrid(0:0.1:10),1) == 101

        @test collect(EquidistantGrid(0:0.1:0.5)) == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        @test Base.IteratorSize(EquidistantGrid{Float64, StepRange{Float64}}) == Base.HasShape{1}()
    end

    @testset "Base" begin
        @test ndims(EquidistantGrid(0:10)) == 1

        g = EquidistantGrid(0:0.1:10)
        @test axes(g,1) == 1:101
        @test axes(g) == (1:101,)
    end

    @testset "spacing" begin
        @test spacing(EquidistantGrid(0:10)) == 1
        @test spacing(EquidistantGrid(0:0.1:10)) == 0.1
    end

    @testset "inverse_spacing" begin
        @test inverse_spacing(EquidistantGrid(0:10)) == 1
        @test inverse_spacing(EquidistantGrid(0:0.1:10)) == 10
    end

    @testset "min_spacing" begin
        @test min_spacing(EquidistantGrid(0:10)) == 1
        @test min_spacing(EquidistantGrid(0:0.1:10)) == 0.1
    end

    @testset "boundary_identifiers" begin
        g = EquidistantGrid(0:0.1:10)
        @test boundary_identifiers(g) == (LowerBoundary(), UpperBoundary())
        @inferred boundary_identifiers(g)
    end

    @testset "boundary_grid" begin
        g = EquidistantGrid(0:0.1:1)
        @test boundary_grid(g, LowerBoundary()) == ZeroDimGrid(0.0)
        @test boundary_grid(g, UpperBoundary()) == ZeroDimGrid(1.0)
    end

    @testset "boundary_indices" begin
        g = EquidistantGrid(0:0.1:1)
        @test boundary_indices(g, LowerBoundary()) == (1,)
        @test boundary_indices(g, UpperBoundary()) == (11,)

        g = EquidistantGrid(2:0.1:10)
        @test boundary_indices(g, LowerBoundary()) == (1,)
        @test boundary_indices(g, UpperBoundary()) == (81,)

    end

    @testset "refine" begin
        g = EquidistantGrid(0:0.1:1)
        @test refine(g, 1) == g
        @test refine(g, 2) == EquidistantGrid(0:0.05:1)
        @test refine(g, 3) == EquidistantGrid(0:(0.1/3):1)
    end

    @testset "coarsen" begin
        g = EquidistantGrid(0:1:10)
        @test coarsen(g, 1) == g
        @test coarsen(g, 2) == EquidistantGrid(0:2:10)

        g = EquidistantGrid(0:0.1:1)
        @test coarsen(g, 1) == g
        @test coarsen(g, 2) == EquidistantGrid(0:0.2:1)

        g = EquidistantGrid(0:10)
        @test coarsen(g, 1) == EquidistantGrid(0:1:10)
        @test coarsen(g, 2) == EquidistantGrid(0:2:10)

        @test_throws DomainError(3, "Size minus 1 must be divisible by the ratio.") coarsen(g, 3)
    end
end


@testset "equidistant_grid" begin
    @test equidistant_grid(0.0,1.0, 4) isa EquidistantGrid
    @test equidistant_grid((0.0,0.0),(8.0,5.0), 4, 3) isa TensorGrid
    @test equidistant_grid((0.0,),(8.0,), 4) isa TensorGrid

    # constuctor
    @test_throws DomainError equidistant_grid(0.0, 1.0, 0)
    @test_throws DomainError equidistant_grid(1.0, 1.0, 1)
    @test_throws DomainError equidistant_grid(1.0, -1.0, 1)

    @test_throws DomainError equidistant_grid((0.0,0.0),(1.0,1.0), 0, 0)
    @test_throws DomainError equidistant_grid((1.0,1.0),(1.0,1.0), 1, 1)
    @test_throws DomainError equidistant_grid((1.0,1.0),(-1.0,-1.0), 1, 1)

    @test_throws ArgumentError equidistant_grid((0.0,),(8.0,5.0), 4, 3, 4)

    @testset "Base" begin
        @test eltype(equidistant_grid(0.0, 1.0, 4)) == Float64
        @test eltype(equidistant_grid((0,0),(1,3), 4, 3)) <: AbstractVector{Float64}

        @test size(equidistant_grid(0.0, 1.0, 4)) == (4,)
        @test size(equidistant_grid((0.0,0.0), (2.0,1.0), 5, 3)) == (5,3)

        @test size(equidistant_grid((0.0,0.0), (2.0,1.0), 5, 3), 1) == 5
        @test size(equidistant_grid((0.0,0.0), (2.0,1.0), 5, 3), 2) == 3

        @test ndims(equidistant_grid(0.0, 1.0, 4)) == 1
        @test ndims(equidistant_grid((0.0,0.0), (2.0,1.0), 5, 3)) == 2
    end

    @testset "getindex" begin
        g = equidistant_grid((-1.0,0.0), (0.0,7.11), 5, 3)
        gp = collect(g);
        p = [(-1.,0.)      (-1.,7.11/2)   (-1.,7.11);
            (-0.75,0.)    (-0.75,7.11/2) (-0.75,7.11);
            (-0.5,0.)     (-0.5,7.11/2)  (-0.5,7.11);
            (-0.25,0.)    (-0.25,7.11/2) (-0.25,7.11);
            (0.,0.)       (0.,7.11/2)    (0.,7.11)]
        for i ∈ eachindex(gp)
            @test [gp[i]...] ≈ [p[i]...] atol=5e-13
        end
    end


    @testset "equidistant_grid(::ParameterSpace)" begin
        ps = HyperBox((0,0),(2,1))

        @test equidistant_grid(ps, 3,4) == equidistant_grid((0,0), (2,1), 3,4)
    end


    @testset "equidistant_grid(::Chart)" begin
        c = Chart(unitsquare()) do (ξ,η)
            @SVector[2ξ, 3η]
        end
        Grids.jacobian(c::typeof(c), ξ̄) = @SMatrix[2 0; 0 3]

        @test equidistant_grid(c, 5, 4) isa Grid
    end
end


@testset "change_length" begin
    @test Grids.change_length(0:20, 21) == 0:20
    @test Grids.change_length(0:20, 11) == 0:2:20
    @test Grids.change_length(0:2:20, 21) == 0:20

    @test Grids.change_length(range(0,1,length=10), 10) == range(0,1,length=10)
    @test Grids.change_length(range(0,1,length=10), 5) == range(0,1,length=5)
    @test Grids.change_length(range(0,1,length=10), 20) == range(0,1,length=20)

    @test Grids.change_length(LinRange(1,2,10),10) == LinRange(1,2,10)
    @test Grids.change_length(LinRange(1,2,10),15) == LinRange(1,2,15)
end
