include("test_utils.jl")
using Test
using TestSetExtensions

@testset "All" begin
    @includetests ARGS
end
