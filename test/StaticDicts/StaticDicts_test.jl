using Test
using Sbplib.StaticDicts

@testset "StaticDicts" begin

@testset "StaticDict" begin
    @testset "constructor" begin
        @test (StaticDict{Int,Int,N} where N) <: AbstractDict

        d = StaticDict(1=>2, 3=>4)
        @test d isa StaticDict{Int,Int}
        @test d[1] == 2
        @test d[3] == 4

        @test StaticDict((1=>2, 3=>4)) == d

        @test StaticDict() isa StaticDict
        @test StaticDict{Int,String}() isa StaticDict{Int,String,0}

        @test StaticDict(1=>3, 2=>4.) isa StaticDict{Int,Real}
        @test StaticDict(1. =>3, 2=>4) isa StaticDict{Real,Int}
        @test StaticDict(1. =>3, 2=>4.) isa StaticDict{Real,Real}

        @test_throws DomainError StaticDict(1=>3, 1=>3)
    end

    @testset "length" begin
        @test length(StaticDict()) == 0
        @test length(StaticDict(1=>1)) == 1
        @test length(StaticDict(1=>1, 2=>2)) == 2
    end

    @testset "equality" begin
        @test StaticDict(1=>1) == StaticDict(1=>1)
        @test StaticDict(2=>1) != StaticDict(1=>1)
        @test StaticDict(1=>2) != StaticDict(1=>1)

        @test StaticDict(1=>1) === StaticDict(1=>1) #not true for a regular Dict
        @test StaticDict(2=>1) !== StaticDict(1=>1)
        @test StaticDict(1=>2) !== StaticDict(1=>1)
    end

    @testset "get" begin
        d = StaticDict(1=>2, 3=>4)

        @test get(d,1,6) == 2
        @test get(d,3,6) == 4
        @test get(d,5,6) == 6
    end

    @testset "iterate" begin
        pairs = [1=>2, 3=>4, 5=>6]

        d = StaticDict(pairs...)
        @test collect(d) == pairs
    end

    @testset "merge" begin
        @test merge(
            StaticDict(1=>3, 2=> 4),
            StaticDict(3=>5,4=>6)) == StaticDict(
                1=>3, 2=>4, 3=>5, 4=>6
            )
        @test_throws DomainError merge(StaticDict(1=>3),StaticDict(1=>3))
    end
end

end
