using Test
using LazyTensors
using RegionIndices

@testset "Generic Mapping methods" begin
    struct DummyMapping{T,R,D} <: TensorMapping{T,R,D} end
    LazyTensors.apply(m::DummyMapping{T,R,D}, v, i::NTuple{R,Index{<:Region}}) where {T,R,D} = :apply
    @test range_dim(DummyMapping{Int,2,3}()) == 2
    @test domain_dim(DummyMapping{Int,2,3}()) == 3
    @test apply(DummyMapping{Int,2,3}(), zeros(Int, (0,0,0)),(Index{Unknown}(0),Index{Unknown}(0))) == :apply
end

@testset "Generic Operator methods" begin
    struct DummyOperator{T,D} <: TensorOperator{T,D} end
    @test range_size(DummyOperator{Int,2}(), (3,5)) == (3,5)
    @test domain_size(DummyOperator{Float64, 3}(), (3,3,1)) == (3,3,1)
end

@testset "Mapping transpose" begin
    struct DummyMapping{T,R,D} <: TensorMapping{T,R,D} end

    LazyTensors.apply(m::DummyMapping{T,R,D}, v, I::Vararg{Index{<:Region},R}) where {T,R,D} = :apply
    LazyTensors.apply_transpose(m::DummyMapping{T,R,D}, v, I::Vararg{Index{<:Region},D}) where {T,R,D} = :apply_transpose

    LazyTensors.range_size(m::DummyMapping{T,R,D}, domain_size::NTuple{D,Integer}) where {T,R,D} = :range_size
    LazyTensors.domain_size(m::DummyMapping{T,R,D}, range_size::NTuple{R,Integer}) where {T,R,D} = :domain_size

    m = DummyMapping{Float64,2,3}()
    I = Index{Unknown}(0)
    @test m' isa TensorMapping{Float64, 3,2}
    @test m'' == m
    @test apply(m',zeros(Float64,(0,0)), I, I, I) == :apply_transpose
    @test apply(m'',zeros(Float64,(0,0,0)), I, I) == :apply
    @test apply_transpose(m', zeros(Float64,(0,0,0)), I, I) == :apply

    @test range_size(m', (0,0)) == :domain_size
    @test domain_size(m', (0,0,0)) == :range_size
end

@testset "TensorApplication" begin
    struct DummyMapping{T,R,D} <: TensorMapping{T,R,D} end

    LazyTensors.apply(m::DummyMapping{T,R,D}, v, i::Vararg{Index{<:Region},R}) where {T,R,D} = (:apply,v,i)
    LazyTensors.range_size(m::DummyMapping{T,R,D}, domain_size::NTuple{D,Integer}) where {T,R,D} = 2 .* domain_size
    LazyTensors.domain_size(m::DummyMapping{T,R,D}, range_size::NTuple{R,Integer}) where {T,R,D} = range_size.÷2


    m = DummyMapping{Int, 1, 1}()
    v = [0,1,2]
    @test m*v isa AbstractVector{Int}
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[Index{Upper}(0)] == (:apply,v,(Index{Upper}(0),))
    @test (m*v)[0] == (:apply,v,(Index{Unknown}(0),))
    @test m*m*v isa AbstractVector{Int}
    @test (m*m*v)[Index{Upper}(1)] == (:apply,m*v,(Index{Upper}(1),))
    @test (m*m*v)[1] == (:apply,m*v,(Index{Unknown}(1),))
    @test (m*m*v)[Index{Interior}(3)] == (:apply,m*v,(Index{Interior}(3),))
    @test (m*m*v)[3] == (:apply,m*v,(Index{Unknown}(3),))
    @test (m*m*v)[Index{Lower}(6)] == (:apply,m*v,(Index{Lower}(6),))
    @test (m*m*v)[6] == (:apply,m*v,(Index{Unknown}(6),))
    @test_broken BoundsError == (m*m*v)[0]
    @test_broken BoundsError == (m*m*v)[7]

    m = DummyMapping{Int, 2, 1}()
    @test_throws MethodError m*ones(Int,2,2)
    @test_throws MethodError m*m*v

    m = DummyMapping{Float64, 2, 2}()
    v = ones(3,3)
    I = (Index{Lower}(1),Index{Interior}(2));
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[I] == (:apply,v,I)

    struct ScalingOperator{T,D} <: TensorOperator{T,D}
        λ::T
    end

    LazyTensors.apply(m::ScalingOperator{T,D}, v, I::Vararg{Index,D}) where {T,D} = m.λ*v[I]

    m = ScalingOperator{Int,1}(2)
    v = [1,2,3]
    @test m*v isa AbstractVector
    @test m*v == [2,4,6]

    m = ScalingOperator{Int,2}(2)
    v = [[1 2];[3 4]]
    @test m*v == [[2 4];[6 8]]
    I = (Index{Upper}(2),Index{Lower}(1))
    @test (m*v)[I] == 6
end

@testset "TensorMapping binary operations" begin
    struct ScalarMapping{T,R,D} <: TensorMapping{T,R,D}
        λ::T
    end

    LazyTensors.apply(m::ScalarMapping{T,R,D}, v, I::Vararg{Index{<:Region}}) where {T,R,D} = m.λ*v[I...]
    LazyTensors.range_size(m::ScalarMapping, domain_size) = domain_size
    LazyTensors.domain_size(m::ScalarMapping, range_sizes) = range_sizes

    A = ScalarMapping{Float64,1,1}(2.0)
    B = ScalarMapping{Float64,1,1}(3.0)

    v = [1.1,1.2,1.3]
    for i ∈ eachindex(v)
        @test ((A+B)*v)[i] == 2*v[i] + 3*v[i]
    end

    for i ∈ eachindex(v)
        @test ((A-B)*v)[i] == 2*v[i] - 3*v[i]
    end

    @test range_size(A+B, (3,)) == range_size(A, (3,)) == range_size(B,(3,))
    @test domain_size(A+B, (3,)) == domain_size(A, (3,)) == domain_size(B,(3,))
end

@testset "LazyArray" begin
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
    @test_throws DimensionMismatch v1 + v2
end
