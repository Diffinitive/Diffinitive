using Test
using Diffinitive.LazyTensors

@testset "Generic Mapping methods" begin
    struct DummyMapping{T,R,D} <: LazyTensor{T,R,D} end
    LazyTensors.apply(m::DummyMapping{T,R,D}, v, I::Vararg{Any,R}) where {T,R,D} = :apply
    @test range_dim(DummyMapping{Int,2,3}()) == 2
    @test domain_dim(DummyMapping{Int,2,3}()) == 3
    @test apply(DummyMapping{Int,2,3}(), zeros(Int, (0,0,0)),0,0) == :apply
    @test eltype(DummyMapping{Int,2,3}()) == Int
    @test eltype(DummyMapping{Float64,2,3}()) == Float64
end
