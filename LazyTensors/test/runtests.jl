using Test
using LazyTensors

@testset "Generic Mapping methods" begin
    struct DummyMapping{T,R,D} <: TensorMapping{T,R,D} end
    LazyTensors.apply(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = :apply
    @test range_dim(DummyMapping{Int,2,3}()) == 2
    @test domain_dim(DummyMapping{Int,2,3}()) == 3
    @test apply(DummyMapping{Int,2,3}(), zeros(Int, (0,0,0)),0) == :apply
end

@testset "Generic Operator methods" begin
    struct DummyOperator{T,D} <: TensorOperator{T,D} end
    @test range_size(DummyOperator{Int,2}(), (3,5)) == (3,5)
    @test domain_size(DummyOperator{Float64, 3}(), (3,3,1)) == (3,3,1)
end

@testset "Mapping transpose" begin
    struct DummyMapping{T,R,D} <: TensorMapping{T,R,D} end

    LazyTensors.apply(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = :apply
    LazyTensors.apply_transpose(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = :apply_transpose

    LazyTensors.range_size(m::DummyMapping{T,R,D}, domain_size) where {T,R,D} = :range_size
    LazyTensors.domain_size(m::DummyMapping{T,R,D}, range_size) where {T,R,D} = :domain_size

    m = DummyMapping{Float64,2,3}()
    @test m'' == m
    @test apply(m',zeros(Float64,(0,0)),0) == :apply_transpose
    @test apply(m'',zeros(Float64,(0,0,0)),0) == :apply
    @test apply_transpose(m', zeros(Float64,(0,0,0)),0) == :apply

    @test range_size(m', (0,0)) == :domain_size
    @test domain_size(m', (0,0,0)) == :range_size
end

@testset "TensorApplication" begin
    struct DummyMapping{T,R,D} <: TensorMapping{T,R,D} end

    LazyTensors.apply(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = (:apply,v,i)
    LazyTensors.apply_transpose(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = :apply_transpose

    LazyTensors.range_size(m::DummyMapping{T,R,D}, domain_size) where {T,R,D} = 2 .* domain_size
    LazyTensors.domain_size(m::DummyMapping{T,R,D}, range_size) where {T,R,D} = range_size.÷2


    m = DummyMapping{Int, 1, 1}()
    v = [0,1,2]
    @test m*v isa AbstractVector{Int}
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[0] == (:apply,v,0)
    @test m*m*v isa AbstractVector{Int}
    @test (m*m*v)[1] == (:apply,m*v,1)
    @test (m*m*v)[3] == (:apply,m*v,3)
    @test (m*m*v)[6] == (:apply,m*v,6)
    @test_broken BoundsError == (m*m*v)[0]
    @test_broken BoundsError == (m*m*v)[7]
end

@testset "LazyArray" begin
    struct DummyArray{T,D, T1<:AbstractArray{T,D}} <: LazyArray{T,D}
        data::T1
    end
    Base.size(v::DummyArray) = size(v.data)
    Base.getindex(v::DummyArray, I...) = v.data[I...]

    # Test lazy operations
    v1 = [1, 2.3, 4]
    v2 = [1., 2, 3]
    r_add = v1 .+ v2
    r_sub = v1 .- v2
    r_times = v1 .* v2
    r_div = v1 ./ v2
    @test isa(v1 +̃ v2, LazyArray)
    @test isa(v1 -̃ v2, LazyArray)
    @test isa(v1 *̃ v2, LazyArray)
    @test isa(v1 /̃ v2, LazyArray)
    for i ∈ eachindex(v1)
        @test (v1 +̃ v2)[i] == r_add[i]
        @test (v1 -̃ v2)[i] == r_sub[i]
        @test (v1 *̃ v2)[i] == r_times[i]
        @test (v1 /̃ v2)[i] == r_div[i]
    end
    @test_throws BoundsError (v1 +̃  v2)[4]
    v2 = [1., 2, 3, 4]
    # Test that size of arrays is asserted when not specified inbounds
    @test_throws DimensionMismatch v1 +̃ v2
    # Test that no error checking is performed when specified inbounds
    res = (v1,v2) -> (@inbounds (v1 +̃ v2)[1] == 2)
    @test res(v1,v2)

    # Test operations on LazyArray
    v1 = DummyArray([1, 2.3, 4])
    v2 = [1., 2, 3]
    @test isa(v1 + v2, LazyArray)
    @test isa(v2 + v1, LazyArray)
    @test isa(v1 - v2, LazyArray)
    @test isa(v2 - v1, LazyArray)
    for i ∈ eachindex(v2)
        @test (v1 + v2)[i] == (v2 + v1)[i] == r_add[i]
        @test (v1 - v2)[i] == -(v2 - v1)[i] == r_sub[i]
    end
    @test_throws BoundsError (v1 + v2)[4]
    v2 = [1., 2, 3, 4]
    # Test that size of arrays is asserted when not specified inbounds
    @test_throws DimensionMismatch v1 + v2
    # Test that no error checking is performed when specified inbounds
    res = (v1,v2) -> (@inbounds (v1 + v2)[1] == 2)
    @test res(v1,v2)
end
