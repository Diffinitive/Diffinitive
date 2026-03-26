using Test
using Diffinitive.LazyTensors

@testset "Generic Mapping methods" begin
    struct DummyMapping{R,D} <: LazyTensor{R,D} end
    LazyTensors.apply(m::DummyMapping{R,D}, v, I::Vararg{Any,R}) where {R,D} = :apply
    @test range_dim(DummyMapping{2,3}()) == 2
    @test domain_dim(DummyMapping{2,3}()) == 3
    @test apply(DummyMapping{2,3}(), zeros(Int, (0,0,0)),0,0) == :apply
end
