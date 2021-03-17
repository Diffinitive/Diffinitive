using Test

using TOML
using Sbplib.SbpOperators

import Sbplib.SbpOperators.Stencil


@testset "parse_rational" begin
    @test SbpOperators.parse_rational("1") isa Rational
    @test SbpOperators.parse_rational("1") == 1//1
    @test SbpOperators.parse_rational("1/2") isa Rational
    @test SbpOperators.parse_rational("1/2") == 1//2
    @test SbpOperators.parse_rational("37/13") isa Rational
    @test SbpOperators.parse_rational("37/13") == 37//13
end

@testset "readoperator" begin
    toml_str = """
        [meta]
        type = "equidistant"

        [order2]
        H.inner = ["1"]

        D1.inner_stencil = ["-1/2", "0", "1/2"]
        D1.closure_stencils = [
            ["-1", "1"],
        ]

        d1.closure = ["-3/2", "2", "-1/2"]

        [order4]
        H.closure = ["17/48", "59/48", "43/48", "49/48"]

        D2.inner_stencil = ["-1/12","4/3","-5/2","4/3","-1/12"]
        D2.closure_stencils = [
            [     "2",    "-5",      "4",       "-1",     "0",     "0"],
            [     "1",    "-2",      "1",        "0",     "0",     "0"],
            [ "-4/43", "59/43", "-110/43",   "59/43", "-4/43",     "0"],
            [ "-1/49",     "0",   "59/49", "-118/49", "64/49", "-4/49"],
        ]
    """

    parsed_toml = TOML.parse(toml_str)
    @testset "get_stencil" begin
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil") == Stencil(-1/2, 0., 1/2, center=2)
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil", center=1) == Stencil(-1/2, 0., 1/2; center=1)
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil", center=3) == Stencil(-1/2, 0., 1/2; center=3)

        @test get_stencil(parsed_toml, "order2", "H", "inner") == Stencil(1.; center=1)

        @test_throws AssertionError get_stencil(parsed_toml, "meta", "type")
        @test_throws AssertionError get_stencil(parsed_toml, "order2", "D1", "closure_stencils")
    end

    @testset "get_stencils" begin
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=(1,)) == (Stencil(-1., 1., center=1),)
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=(2,)) == (Stencil(-1., 1., center=2),)
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=[2]) == (Stencil(-1., 1., center=2),)

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=[1,1,1,1]) == (
            Stencil(    2.,    -5.,      4.,     -1.,    0.,    0., center=1),
            Stencil(    1.,    -2.,      1.,      0.,    0.,    0., center=1),
            Stencil( -4/43,  59/43, -110/43,   59/43, -4/43,    0., center=1),
            Stencil( -1/49,     0.,   59/49, -118/49, 64/49, -4/49, center=1),
        )

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=(4,2,3,1)) == (
            Stencil(    2.,    -5.,      4.,     -1.,    0.,    0., center=4),
            Stencil(    1.,    -2.,      1.,      0.,    0.,    0., center=2),
            Stencil( -4/43,  59/43, -110/43,   59/43, -4/43,    0., center=3),
            Stencil( -1/49,     0.,   59/49, -118/49, 64/49, -4/49, center=1),
        )

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=1:4) == (
            Stencil(    2.,    -5.,      4.,     -1.,    0.,    0., center=1),
            Stencil(    1.,    -2.,      1.,      0.,    0.,    0., center=2),
            Stencil( -4/43,  59/43, -110/43,   59/43, -4/43,    0., center=3),
            Stencil( -1/49,     0.,   59/49, -118/49, 64/49, -4/49, center=4),
        )

        @test_throws AssertionError get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=(1,2,3))
        @test_throws AssertionError get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=(1,2,3,5,4))
        @test_throws AssertionError get_stencils(parsed_toml, "order4", "D2", "inner_stencil",centers=(1,2))
    end

    @testset "get_tuple" begin
        @test get_tuple(parsed_toml, "order2", "d1", "closure") == (-3/2, 2, -1/2)

        @test_throws AssertionError get_tuple(parsed_toml, "meta", "type")
    end
end
