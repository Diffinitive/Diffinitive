using Sbplib.Grids
using Test
using Sbplib.RegionIndices
using Sbplib.LazyTensors


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
    end

    @testset "Iterator interface" begin
        @test eltype(EquidistantGrid(0:10)) == Int
        @test eltype(EquidistantGrid(0:2:10)) == Int
        @test eltype(EquidistantGrid(0:0.1:10)) == Float64
        @test size(EquidistantGrid(0:10)) == (11,)
        @test size(EquidistantGrid(0:0.1:10)) == (101,)

        @test collect(EquidistantGrid(0:0.1:0.5)) == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        @test Base.IteratorSize(EquidistantGrid{Float64, StepRange{Float64}}) == Base.HasShape{1}()
    end

    @testset "Base" begin
        @test ndims(EquidistantGrid(0:10)) == 1
    end

    @testset "spacing" begin
        @test spacing(EquidistantGrid(0:10)) == 1
        @test spacing(EquidistantGrid(0:0.1:10)) == 0.1
    end

    @testset "inverse_spacing" begin
        @test inverse_spacing(EquidistantGrid(0:10)) == 1
        @test inverse_spacing(EquidistantGrid(0:0.1:10)) == 10
    end

    @testset "boundary_identifiers" begin
        g = EquidistantGrid(0:0.1:10)
        @test boundary_identifiers(g) == (Lower(), Upper())
        @inferred boundary_identifiers(g)
    end

    @testset "boundary_grid" begin
        g = EquidistantGrid(0:0.1:1)
        @test boundary_grid(g, Lower()) == ZeroDimGrid(0.0) # TBD: Is fill necessary here? Why?
        @test boundary_grid(g, Upper()) == ZeroDimGrid(1.0)
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
    @test equidistant_grid(4,0.0,1.0) isa TensorGrid
    @test equidistant_grid(4,0.0,8.0) isa TensorGrid
    # constuctor
    @test_throws DomainError equidistant_grid(0,0.0,1.0)
    @test_throws DomainError equidistant_grid(1,1.0,1.0)
    @test_throws DomainError equidistant_grid(1,1.0,-1.0)
    @test equidistant_grid(4,0.0,1.0) == equidistant_grid((4,),(0.0,),(1.0,))

    @testset "Base" begin
        @test eltype(equidistant_grid(4,0.0,1.0)) == Float64
        @test eltype(equidistant_grid((4,3),(0,0),(1,3))) <: AbstractVector{Float64}
        @test size(equidistant_grid(4,0.0,1.0)) == (4,)
        @test size(equidistant_grid((5,3), (0.0,0.0), (2.0,1.0))) == (5,3)
        @test ndims(equidistant_grid(4,0.0,1.0)) == 1
        @test ndims(equidistant_grid((5,3), (0.0,0.0), (2.0,1.0))) == 2
    end

    @testset "getindex" begin
        g = equidistant_grid((5,3), (-1.0,0.0), (0.0,7.11))
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
end
