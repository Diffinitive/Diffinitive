using Test
using Sbplib.LazyTensors
using Sbplib.RegionIndices


@testset "LazyArray" begin
    @testset "LazyConstantArray" begin
        @test LazyTensors.LazyConstantArray(3,(3,2)) isa LazyArray{Int,2}

        lca = LazyTensors.LazyConstantArray(3.0,(3,2))
        @test eltype(lca) == Float64
        @test ndims(lca) == 2
        @test size(lca) == (3,2)
        @test lca[2] == 3.0
    end
    struct DummyArray{T,D, T1<:AbstractArray{T,D}} <: LazyArray{T,D}
        data::T1
    end
    Base.size(v::DummyArray) = size(v.data)
    Base.getindex(v::DummyArray{T,D}, I::Vararg{Int,D}) where {T,D} = v.data[I...]

    # Test lazy operations
    v1 = [1, 2.3, 4]
    v2 = [1., 2, 3]
    s = 3.4
    r_add_v = v1 .+ v2
    r_sub_v = v1 .- v2
    r_times_v = v1 .* v2
    r_div_v = v1 ./ v2
    r_add_s = v1 .+ s
    r_sub_s = v1 .- s
    r_times_s = v1 .* s
    r_div_s = v1 ./ s
    @test isa(v1 +̃ v2, LazyArray)
    @test isa(v1 -̃ v2, LazyArray)
    @test isa(v1 *̃ v2, LazyArray)
    @test isa(v1 /̃ v2, LazyArray)
    @test isa(v1 +̃ s, LazyArray)
    @test isa(v1 -̃ s, LazyArray)
    @test isa(v1 *̃ s, LazyArray)
    @test isa(v1 /̃ s, LazyArray)
    @test isa(s +̃ v1, LazyArray)
    @test isa(s -̃ v1, LazyArray)
    @test isa(s *̃ v1, LazyArray)
    @test isa(s /̃ v1, LazyArray)
    for i ∈ eachindex(v1)
        @test (v1 +̃ v2)[i] == r_add_v[i]
        @test (v1 -̃ v2)[i] == r_sub_v[i]
        @test (v1 *̃ v2)[i] == r_times_v[i]
        @test (v1 /̃ v2)[i] == r_div_v[i]
        @test (v1 +̃ s)[i] == r_add_s[i]
        @test (v1 -̃ s)[i] == r_sub_s[i]
        @test (v1 *̃ s)[i] == r_times_s[i]
        @test (v1 /̃ s)[i] == r_div_s[i]
        @test (s +̃ v1)[i] == r_add_s[i]
        @test (s -̃ v1)[i] == -r_sub_s[i]
        @test (s *̃ v1)[i] == r_times_s[i]
        @test (s /̃ v1)[i] == 1/r_div_s[i]
    end
    @test_throws BoundsError (v1 +̃  v2)[4]
    v2 = [1., 2, 3, 4]
    # Test that size of arrays is asserted when not specified inbounds
    # TODO: Replace these errors with SizeMismatch
    @test_throws DimensionMismatch v1 +̃ v2

    # Test operations on LazyArray
    v1 = DummyArray([1, 2.3, 4])
    v2 = [1., 2, 3]
    @test isa(v1 + v2, LazyArray)
    @test isa(v2 + v1, LazyArray)
    @test isa(v1 - v2, LazyArray)
    @test isa(v2 - v1, LazyArray)
    for i ∈ eachindex(v2)
        @test (v1 + v2)[i] == (v2 + v1)[i] == r_add_v[i]
        @test (v1 - v2)[i] == -(v2 - v1)[i] == r_sub_v[i]
    end
    @test_throws BoundsError (v1 + v2)[4]
    v2 = [1., 2, 3, 4]
    # Test that size of arrays is asserted when not specified inbounds
    # TODO: Replace these errors with SizeMismatch
    @test_throws DimensionMismatch v1 + v2
end


@testset "LazyFunctionArray" begin
    @test LazyFunctionArray(i->i^2, (3,)) == [1,4,9]
    @test LazyFunctionArray((i,j)->i*j, (3,2)) == [
        1 2;
        2 4;
        3 6;
    ]

    @test size(LazyFunctionArray(i->i^2, (3,))) == (3,)
    @test size(LazyFunctionArray((i,j)->i*j, (3,2))) == (3,2)

    @inferred LazyFunctionArray(i->i^2, (3,))[2]

    @test_throws BoundsError LazyFunctionArray(i->i^2, (3,))[4]
    @test_throws BoundsError LazyFunctionArray((i,j)->i*j, (3,2))[4,2]
    @test_throws BoundsError LazyFunctionArray((i,j)->i*j, (3,2))[2,3]

end
