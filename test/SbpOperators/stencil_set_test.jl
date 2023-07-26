using Test

using TOML
using Sbplib.SbpOperators

import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.NestedStencil

@testset "readoperator" begin
    toml_str = """
        [meta]
        authors = "Ken Mattson"
        description = "Standard operators for equidistant grids"
        type = "equidistant"
        cite = "A paper a long time ago in a galaxy far far away."

        [[stencil_set]]

        order = 2
        test = 2

        H.inner = ["1"]
        H.closure = ["1/2"]

        D1.inner_stencil = ["-1/2", "0", "1/2"]
        D1.closure_stencils = [
            {s = ["-1", "1"], c = 1},
        ]

        D2.inner_stencil = ["1", "-2", "1"]
        D2.closure_stencils = [
            {s = ["1", "-2", "1"], c = 1},
        ]

        e.closure = ["1"]
        d1.closure = {s = ["-3/2", "2", "-1/2"], c = 1}

        [[stencil_set]]

        order = 4
        test = 1
        H.inner = ["1"]
        H.closure = ["17/48", "59/48", "43/48", "49/48"]

        D2.inner_stencil = ["-1/12","4/3","-5/2","4/3","-1/12"]
        D2.closure_stencils = [
            {s = [     "2",    "-5",      "4",       "-1",     "0",     "0"], c = 1},
            {s = [     "1",    "-2",      "1",        "0",     "0",     "0"], c = 2},
            {s = [ "-4/43", "59/43", "-110/43",   "59/43", "-4/43",     "0"], c = 3},
            {s = [ "-1/49",     "0",   "59/49", "-118/49", "64/49", "-4/49"], c = 4},
        ]

        e.closure = ["1"]
        d1.closure = {s = ["-11/6", "3", "-3/2", "1/3"], c = 1}

        [[stencil_set]]
        order = 4
        test = 2

        H.closure = ["-1/49", "0", "59/49", "-118/49", "64/49", "-4/49"]
    """

    parsed_toml = TOML.parse(toml_str)

    @testset "get_stencil_set" begin
        @test get_stencil_set(parsed_toml; order = 2) isa Dict
        @test get_stencil_set(parsed_toml; order = 2) == parsed_toml["stencil_set"][1]
        @test get_stencil_set(parsed_toml; test = 1) == parsed_toml["stencil_set"][2]
        @test get_stencil_set(parsed_toml; order = 4, test = 2) == parsed_toml["stencil_set"][3]

        @test_throws ArgumentError get_stencil_set(parsed_toml; test = 2)
        @test_throws ArgumentError get_stencil_set(parsed_toml; order = 4)
    end

    @testset "parse_stencil" begin
        toml = """
            s1 = ["-1/12","4/3","-5/2","4/3","-1/12"]
            s2 = {s = ["2", "-5", "4", "-1", "0", "0"], c = 1}
            s3 = {s = ["1", "-2", "1", "0", "0", "0"], c = 2}
            s4 = "not a stencil"
            s5 = [-1, 4, 3]
            s6 = {k = ["1", "-2", "1", "0", "0", "0"], c = 2}
            s7 = {s = [-1, 4, 3], c = 2}
            s8 = {s = ["1", "-2", "1", "0", "0", "0"], c = [2,2]}
        """

        @test parse_stencil(TOML.parse(toml)["s1"]) == CenteredStencil(-1//12, 4//3, -5//2, 4//3, -1//12)
        @test parse_stencil(TOML.parse(toml)["s2"]) == Stencil(2//1, -5//1, 4//1, -1//1, 0//1, 0//1; center=1)
        @test parse_stencil(TOML.parse(toml)["s3"]) == Stencil(1//1, -2//1, 1//1, 0//1, 0//1, 0//1; center=2)

        @test_throws ArgumentError parse_stencil(TOML.parse(toml)["s4"])
        @test_throws ArgumentError parse_stencil(TOML.parse(toml)["s5"])
        @test_throws ArgumentError parse_stencil(TOML.parse(toml)["s6"])
        @test_throws ArgumentError parse_stencil(TOML.parse(toml)["s7"])
        @test_throws ArgumentError parse_stencil(TOML.parse(toml)["s8"])

        stencil_set = get_stencil_set(parsed_toml; order = 4, test = 1)

        @test parse_stencil.(stencil_set["D2"]["closure_stencils"]) == [
            Stencil(  2//1,  -5//1,     4//1,    -1//1,   0//1,   0//1; center=1),
            Stencil(  1//1,  -2//1,     1//1,     0//1,   0//1,   0//1; center=2),
            Stencil(-4//43, 59//43, -110//43,   59//43, -4//43,   0//1; center=3),
            Stencil(-1//49,   0//1,   59//49, -118//49, 64//49, -4//49; center=4),
        ]


        @test parse_stencil(Float64, TOML.parse(toml)["s1"]) == CenteredStencil(-1/12, 4/3, -5/2, 4/3, -1/12)
        @test parse_stencil(Float64, TOML.parse(toml)["s2"]) == Stencil(2/1, -5/1, 4/1, -1/1, 0/1, 0/1; center=1)
        @test parse_stencil(Float64, TOML.parse(toml)["s3"]) == Stencil(1/1, -2/1, 1/1, 0/1, 0/1, 0/1; center=2)
    end

    @testset "parse_scalar" begin
        toml = TOML.parse("""
            a1 = 1
            a2 = 1.5
            a3 = 1.0
            a4 = 10
            a5 = "1/2"
            a6 = "1.5"

            e1 = [1,2,3]
            e2 = "a string value"
        """)

        @test parse_scalar(toml["a1"]) == 1//1
        @test parse_scalar(toml["a2"]) == 3//2
        @test parse_scalar(toml["a3"]) == 1//1
        @test parse_scalar(toml["a4"]) == 10//1
        @test parse_scalar(toml["a5"]) == 1//2
        @test parse_scalar(toml["a6"]) == 3//2

        @test_throws ArgumentError parse_scalar(toml["e1"])
        @test_throws ArgumentError parse_scalar(toml["e2"])
    end

    @testset "parse_tuple" begin
        toml = TOML.parse("""
            t1 = [1,3,4]
            t2 = ["1/2","3/4","2/1"]

            e1 = "not a tuple"
            e2.a="1"
            e3 = 1
            e4 = ["1/2","3/4","not a number"]
        """)

        @test parse_tuple(toml["t1"]) == (1//1,3//1,4//1)
        @test parse_tuple(toml["t2"]) == (1//2,3//4,2//1)

        @test_throws ArgumentError parse_tuple(toml["e1"])
        @test_throws ArgumentError parse_tuple(toml["e2"])
        @test_throws ArgumentError parse_tuple(toml["e3"])
        @test_throws ArgumentError parse_tuple(toml["e4"])
    end
end

@testset "parse_rational" begin
    @test SbpOperators.parse_rational("1") isa Rational
    @test SbpOperators.parse_rational("1") == 1//1
    @test SbpOperators.parse_rational("1/2") isa Rational
    @test SbpOperators.parse_rational("1/2") == 1//2
    @test SbpOperators.parse_rational("37/13") isa Rational
    @test SbpOperators.parse_rational("37/13") == 37//13

    @test SbpOperators.parse_rational(0.5) isa Rational
    @test SbpOperators.parse_rational(0.5) == 1//2

    @test SbpOperators.parse_rational("0.5") isa Rational
    @test SbpOperators.parse_rational("0.5") == 1//2

    @test SbpOperators.parse_rational(2) isa Rational
    @test SbpOperators.parse_rational(2) == 2//1
end

@testset "parse_nested_stencil" begin
    toml = TOML.parse("""
        s1 = [["1/2", "1/2", "0"],[ "-1/2", "-1", "-1/2"],["0", "1/2", "1/2"]]
        s2 = {s = [[  "2",  "-1", "0"],[   "-3",  "1",    "0"],["1",   "0",   "0"]], c = 1}
        s3 = {s = [[  "2",  "-1", "0"],[   "-3",  "1",    "0"],["1",   "0",   "0"]], c = 2}
    """)

    @test parse_nested_stencil(toml["s1"]) == CenteredNestedStencil((1//2, 1//2, 0//1),( -1//2, -1//1, -1//2),(0//1, 1//2, 1//2))
    @test parse_nested_stencil(toml["s2"]) == NestedStencil((2//1, -1//1, 0//1),( -3//1, 1//1, 0//1),(1//1, 0//1, 0//1), center = 1)
    @test parse_nested_stencil(toml["s3"]) == NestedStencil((2//1, -1//1, 0//1),( -3//1, 1//1, 0//1),(1//1, 0//1, 0//1), center = 2)

    @test parse_nested_stencil(Float64, toml["s1"]) == CenteredNestedStencil((1/2, 1/2, 0.),( -1/2, -1., -1/2),(0., 1/2, 1/2))
    @test parse_nested_stencil(Int, toml["s2"]) == NestedStencil((2, -1, 0),( -3, 1, 0),(1, 0, 0), center = 1)
end
