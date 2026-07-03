using Test

using Diffinitive.Grids
using Diffinitive.RegionIndices
using Diffinitive.LazyTensors

using ForwardDiff
using StaticArrays

west = CartesianBoundary{1,LowerBoundary}
east = CartesianBoundary{1,UpperBoundary}
south = CartesianBoundary{2,LowerBoundary}
north = CartesianBoundary{2,UpperBoundary}
bottom = CartesianBoundary{3, LowerBoundary}
top = CartesianBoundary{3, UpperBoundary}

@testset "Chart" begin
    X(ξ) = 2ξ
    Grids.jacobian(::typeof(X), ξ) = @SMatrix[2 0; 0 2]
    c = Chart(X, unitsquare())
    @test c isa Chart{2}
    @test parameterspace(c) == unitsquare()
    @test ndims(c) == 2

    @testset "Calling" begin
        c = Chart(X, unitsquare())
        @test c([.3,.2]) == [.6,.4]
        @test_throws DomainError c([3,2])
    end

    @testset "jacobian(::Chart, ⋅)" begin
        c = Chart(X, unitsquare())
        @test_throws DomainError jacobian(c, [3,2])
        @test jacobian(c, [.3,.2]) == [2 0; 0 2]
    end

    @test Set(boundary_identifiers(Chart(X,unitsquare()))) == Set([east(),west(),south(),north()])
end

@testset "with_jacobian" begin
    @testset "Function with jacobian" begin
        x(ξ) = @SVector[ξ[1] + 2ξ[2], ξ[1]*ξ[2]]
        xJ = with_jacobian(x, ForwardDiff.jacobian)

        @test xJ(@SVector[3, 4]) == @SVector[11, 12]
        @test jacobian(xJ, @SVector[3, 4]) == @SMatrix[
            1 2;
            4 3;
        ]
    end

    @testset "Chart with jacobian" begin
        x(ξ) = @SVector[ξ[1]^2, ξ[1] + ξ[2]]

        c = with_jacobian(x, unitsquare(), ForwardDiff.jacobian)
        x = @SVector[1//2, 1//4]

        @test c isa Chart{2}
        @test parameterspace(c) == unitsquare()
        @test c(x) == @SVector[1//4, 3//4]
        @test jacobian(c, x) == @SMatrix[
            1//1 0//1;
            1//1 1//1;
        ]
    end

    @testset "Chart from combined mapping and jacobian" begin
        xJ(ξ) = (
            @SVector[ξ[1] - ξ[2], ξ[1]*ξ[2]],
            @SMatrix[
                1     -1;
                ξ[2]  ξ[1];
            ],
        )

        c = with_jacobian(xJ, unitsquare())
        x = @SVector[3//4, 1//4]

        @test c isa Chart{2}
        @test parameterspace(c) == unitsquare()
        @test c(x) == @SVector[1//2, 3//16]
        @test jacobian(c, x) == @SMatrix[
            1//1  -1//1;
            1//4   3//4;
        ]
    end

    @testset "Missing jacobian function" begin
        x(ξ) = @SVector[ξ[1]^2, ξ[1] + ξ[2]]

        @test_throws ArgumentError with_jacobian(x, unitsquare())
    end
end

@testset "CartesianAtlas" begin
    @testset "Constructors" begin
        c = Chart(identity, unitsquare())
        @test CartesianAtlas([c c; c c]) isa Atlas

        c2 = Chart(x->2x, unitsquare())
        @test CartesianAtlas([c c2; c2 c]) isa CartesianAtlas
        @test CartesianAtlas(@SMatrix[c c; c c]) isa CartesianAtlas
        @test CartesianAtlas(@SMatrix[c c2; c2 c]) isa CartesianAtlas
    end

    @testset "Getters" begin
        c = Chart(identity, unitsquare())
        a = CartesianAtlas([c c; c c])
        @test charts(a) == [c c; c c]
    end

    @testset "connections" begin
        # 2D
        a = CartesianAtlas(fill(Chart(identity, unitsquare()), 2,3))

        @test Set(connections(a)) == Set([
            (MultiBlockBoundary{(1,1), east}(), MultiBlockBoundary{(2,1), west}()),
            (MultiBlockBoundary{(1,1), north}(), MultiBlockBoundary{(1,2), south}()),
            (MultiBlockBoundary{(2,1), north}(), MultiBlockBoundary{(2,2), south}()),
            (MultiBlockBoundary{(1,2), east}(), MultiBlockBoundary{(2,2), west}()),
            (MultiBlockBoundary{(1,2), north}(), MultiBlockBoundary{(1,3), south}()),
            (MultiBlockBoundary{(2,2), north}(), MultiBlockBoundary{(2,3), south}()),
            (MultiBlockBoundary{(1,3), east}(), MultiBlockBoundary{(2,3), west}()),
        ])

        # 3D
        a = CartesianAtlas(fill(Chart(identity, unitcube()), 2,2,3))
        @test Set(connections(a)) == Set([
            (MultiBlockBoundary{(1,1,1), east}(),  MultiBlockBoundary{(2,1,1), west}()),
            (MultiBlockBoundary{(1,1,1), north}(), MultiBlockBoundary{(1,2,1), south}()),
            (MultiBlockBoundary{(2,1,1), north}(), MultiBlockBoundary{(2,2,1), south}()),
            (MultiBlockBoundary{(1,2,1), east}(),  MultiBlockBoundary{(2,2,1), west}()),

            (MultiBlockBoundary{(1,1,2), east}(),  MultiBlockBoundary{(2,1,2), west}()),
            (MultiBlockBoundary{(1,1,2), north}(), MultiBlockBoundary{(1,2,2), south}()),
            (MultiBlockBoundary{(2,1,2), north}(), MultiBlockBoundary{(2,2,2), south}()),
            (MultiBlockBoundary{(1,2,2), east}(),  MultiBlockBoundary{(2,2,2), west}()),

            (MultiBlockBoundary{(1,1,3), east}(),  MultiBlockBoundary{(2,1,3), west}()),
            (MultiBlockBoundary{(1,1,3), north}(), MultiBlockBoundary{(1,2,3), south}()),
            (MultiBlockBoundary{(2,1,3), north}(), MultiBlockBoundary{(2,2,3), south}()),
            (MultiBlockBoundary{(1,2,3), east}(),  MultiBlockBoundary{(2,2,3), west}()),

            (MultiBlockBoundary{(1,1,1), top}(), MultiBlockBoundary{(1,1,2), bottom}()),
            (MultiBlockBoundary{(2,1,1), top}(), MultiBlockBoundary{(2,1,2), bottom}()),
            (MultiBlockBoundary{(1,2,1), top}(), MultiBlockBoundary{(1,2,2), bottom}()),
            (MultiBlockBoundary{(2,2,1), top}(), MultiBlockBoundary{(2,2,2), bottom}()),

            (MultiBlockBoundary{(1,1,2), top}(), MultiBlockBoundary{(1,1,3), bottom}()),
            (MultiBlockBoundary{(2,1,2), top}(), MultiBlockBoundary{(2,1,3), bottom}()),
            (MultiBlockBoundary{(1,2,2), top}(), MultiBlockBoundary{(1,2,3), bottom}()),
            (MultiBlockBoundary{(2,2,2), top}(), MultiBlockBoundary{(2,2,3), bottom}()),
        ])
    end

    @testset "boundary_identifiers" begin
        # 2D
        a = CartesianAtlas(fill(Chart(identity, unitcube()), 2,3))
        @test Set(boundary_identifiers(a)) == Set([
            MultiBlockBoundary{(1,1), south}(),
            MultiBlockBoundary{(2,1), south}(),
            MultiBlockBoundary{(2,1), east}(),
            MultiBlockBoundary{(2,2), east}(),
            MultiBlockBoundary{(2,3), east}(),
            MultiBlockBoundary{(1,3), north}(),
            MultiBlockBoundary{(2,3), north}(),
            MultiBlockBoundary{(1,1), west}(),
            MultiBlockBoundary{(1,2), west}(),
            MultiBlockBoundary{(1,3), west}(),
        ])

        # 3D
        a = CartesianAtlas(fill(Chart(identity, unitsquare()), 2,2,3))
        @test Set(boundary_identifiers(a)) == Set([
            MultiBlockBoundary{(1,1,1), bottom}(),
            MultiBlockBoundary{(2,1,1), bottom}(),
            MultiBlockBoundary{(1,2,1), bottom}(),
            MultiBlockBoundary{(2,2,1), bottom}(),

            MultiBlockBoundary{(1,1,3), top}(),
            MultiBlockBoundary{(2,1,3), top}(),
            MultiBlockBoundary{(1,2,3), top}(),
            MultiBlockBoundary{(2,2,3), top}(),

            MultiBlockBoundary{(1,1,1), west}(),
            MultiBlockBoundary{(1,2,1), west}(),
            MultiBlockBoundary{(1,1,2), west}(),
            MultiBlockBoundary{(1,2,2), west}(),
            MultiBlockBoundary{(1,1,3), west}(),
            MultiBlockBoundary{(1,2,3), west}(),

            MultiBlockBoundary{(2,1,1), east}(),
            MultiBlockBoundary{(2,2,1), east}(),
            MultiBlockBoundary{(2,1,2), east}(),
            MultiBlockBoundary{(2,2,2), east}(),
            MultiBlockBoundary{(2,1,3), east}(),
            MultiBlockBoundary{(2,2,3), east}(),

            MultiBlockBoundary{(1,1,1), south}(),
            MultiBlockBoundary{(2,1,1), south}(),
            MultiBlockBoundary{(1,1,2), south}(),
            MultiBlockBoundary{(2,1,2), south}(),
            MultiBlockBoundary{(1,1,3), south}(),
            MultiBlockBoundary{(2,1,3), south}(),

            MultiBlockBoundary{(1,2,1), north}(),
            MultiBlockBoundary{(2,2,1), north}(),
            MultiBlockBoundary{(1,2,2), north}(),
            MultiBlockBoundary{(2,2,2), north}(),
            MultiBlockBoundary{(1,2,3), north}(),
            MultiBlockBoundary{(2,2,3), north}(),
        ])
    end
end

@testset "UnstructuredAtlas" begin
    @testset "Constructors" begin
        c1 = Chart(identity, unitsquare())
        c2 = Chart(x->2x, unitsquare())
        cn = [
            (MultiBlockBoundary{1, east}(), MultiBlockBoundary{2, west}()),
            (MultiBlockBoundary{1, north}(), MultiBlockBoundary{3, west}()),
            (MultiBlockBoundary{2, north}(),  MultiBlockBoundary{3, south}()),
        ]

        @test UnstructuredAtlas([c1, c1, c1], cn) isa UnstructuredAtlas
        @test UnstructuredAtlas([c1, c2, c1, c2], cn) isa UnstructuredAtlas


        cn = @SVector[
            (MultiBlockBoundary{1, east}(), MultiBlockBoundary{2, west}()),
            (MultiBlockBoundary{1, north}(), MultiBlockBoundary{3, west}()),
            (MultiBlockBoundary{2, north}(),  MultiBlockBoundary{3, south}()),
        ]
        @test UnstructuredAtlas(@SVector[c1, c1, c1], cn) isa UnstructuredAtlas
        @test UnstructuredAtlas(@SVector[c1, c2, c1, c2], cn) isa UnstructuredAtlas
    end

    @testset "Getters" begin
        c = Chart(identity, unitsquare())
        cn = [
            (MultiBlockBoundary{1, east}(), MultiBlockBoundary{2, west}()),
            (MultiBlockBoundary{1, north}(), MultiBlockBoundary{3, west}()),
            (MultiBlockBoundary{2, north}(),  MultiBlockBoundary{3, south}()),
        ]

        a = UnstructuredAtlas([c, c, c], cn)

        @test charts(a) == [c,c,c]
        @test connections(a) == cn
    end

    @testset "boundary_identifiers" begin
        c = Chart(identity, unitsquare())
        cn = [
            (MultiBlockBoundary{1, east}(), MultiBlockBoundary{2, west}()),
            (MultiBlockBoundary{1, north}(), MultiBlockBoundary{3, west}()),
            (MultiBlockBoundary{2, north}(),  MultiBlockBoundary{3, south}()),
        ]

        a = UnstructuredAtlas([c, c, c], cn)

        @test Set(boundary_identifiers(a)) == Set([
            MultiBlockBoundary{1, west}(),
            MultiBlockBoundary{1, south}(),
            MultiBlockBoundary{2, south}(),
            MultiBlockBoundary{2, east}(),
            MultiBlockBoundary{3, north}(),
            MultiBlockBoundary{3, east}(),
        ])

    end
end
