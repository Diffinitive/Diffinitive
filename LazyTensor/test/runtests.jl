using Test
using LazyTensor



@testset "Generic Mapping methods" begin
    struct DummyMapping{T,R,D} <: LazyTensor.Mapping{T,R,D} end
    LazyTensor.apply(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = :apply
    @test range_dim(DummyMapping{Int,2,3}()) == 2
    @test domain_dim(DummyMapping{Int,2,3}()) == 3
    @test apply(DummyMapping{Int,2,3}(), zeros(Int, (0,0,0)),0) == :apply
end

struct DummyOperator{T,D} <: LazyTensor.Operator{T,D} end
@testset "Generic Operator methods" begin
    @test range_size(DummyOperator{Int,2}(), (3,5)) == (3,5)
    @test domain_size(DummyOperator{Float64, 3}(), (3,3,1)) == (3,3,1)
end

@testset "Mapping transpose" begin
    struct DummyMapping{T,R,D} <: LazyTensor.Mapping{T,R,D} end

    LazyTensor.apply(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = :apply
    LazyTensor.apply_transpose(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = :apply_transpose

    LazyTensor.range_size(m::DummyMapping{T,R,D}, domain_size) where {T,R,D} = :range_size
    LazyTensor.domain_size(m::DummyMapping{T,R,D}, range_size) where {T,R,D} = :domain_size

    m = DummyMapping{Float64,2,3}()
    @test m'' == m
    @test apply(m',zeros(Float64,(0,0)),0) == :apply_transpose
    @test apply(m'',zeros(Float64,(0,0,0)),0) == :apply
    @test apply_transpose(m', zeros(Float64,(0,0,0)),0) == :apply

    @test range_size(m', (0,0)) == :domain_size
    @test domain_size(m', (0,0,0)) == :range_size
end

@testset "TensorApplication" begin
    struct DummyMapping{T,R,D} <: LazyTensor.Mapping{T,R,D} end

    LazyTensor.apply(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = (:apply,v,i)
    LazyTensor.apply_transpose(m::DummyMapping{T,R,D}, v, i) where {T,R,D} = :apply_transpose

    LazyTensor.range_size(m::DummyMapping{T,R,D}, domain_size) where {T,R,D} = 2 .* domain_size
    LazyTensor.domain_size(m::DummyMapping{T,R,D}, range_size) where {T,R,D} = range_size.÷2


    m = DummyMapping{Int, 1, 1}()
    v = [0,1,2]
    @test m*v isa AbstractVector{Int}
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[0] == (:apply,v,0)
    @test m*m*v isa AbstractVector{Int}
    @test (m*m*v)[0] == (:apply,m*v,0)
end