using Test
using Sbplib.LazyTensors
using Sbplib.RegionIndices

using Tullio

@testset "LazyTensors" begin

@testset "Generic Mapping methods" begin
    struct DummyMapping{T,R,D} <: TensorMapping{T,R,D} end
    LazyTensors.apply(m::DummyMapping{T,R,D}, v, i::NTuple{R,Index{<:Region}}) where {T,R,D} = :apply
    @test range_dim(DummyMapping{Int,2,3}()) == 2
    @test domain_dim(DummyMapping{Int,2,3}()) == 3
    @test apply(DummyMapping{Int,2,3}(), zeros(Int, (0,0,0)),(Index{Unknown}(0),Index{Unknown}(0))) == :apply
    @test eltype(DummyMapping{Int,2,3}()) == Int
    @test eltype(DummyMapping{Float64,2,3}()) == Float64
end

@testset "Mapping transpose" begin
    struct DummyMapping{T,R,D} <: TensorMapping{T,R,D} end

    LazyTensors.apply(m::DummyMapping{T,R,D}, v, I::Vararg{Index{<:Region},R}) where {T,R,D} = :apply
    LazyTensors.apply_transpose(m::DummyMapping{T,R,D}, v, I::Vararg{Index{<:Region},D}) where {T,R,D} = :apply_transpose

    LazyTensors.range_size(m::DummyMapping{T,R,D}) where {T,R,D} = :range_size
    LazyTensors.domain_size(m::DummyMapping{T,R,D}) where {T,R,D} = :domain_size

    m = DummyMapping{Float64,2,3}()
    I = Index{Unknown}(0)
    @test m' isa TensorMapping{Float64, 3,2}
    @test m'' == m
    @test apply(m',zeros(Float64,(0,0)), I, I, I) == :apply_transpose
    @test apply(m'',zeros(Float64,(0,0,0)), I, I) == :apply
    @test apply_transpose(m', zeros(Float64,(0,0,0)), I, I) == :apply

    @test range_size(m') == :domain_size
    @test domain_size(m') == :range_size
end

@testset "TensorApplication" begin
    struct SizeDoublingMapping{T,R,D} <: TensorMapping{T,R,D}
        domain_size::NTuple{D,Int}
    end

    LazyTensors.apply(m::SizeDoublingMapping{T,R,D}, v, i::Vararg{Index{<:Region},R}) where {T,R,D} = (:apply,v,i)
    LazyTensors.range_size(m::SizeDoublingMapping) = 2 .* m.domain_size
    LazyTensors.domain_size(m::SizeDoublingMapping) = m.domain_size


    m = SizeDoublingMapping{Int, 1, 1}((3,))
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
    @test_throws MethodError m*m

    m = SizeDoublingMapping{Int, 2, 1}((3,))
    @test_throws MethodError m*ones(Int,2,2)
    @test_throws MethodError m*m*v

    m = SizeDoublingMapping{Float64, 2, 2}((3,3))
    v = ones(3,3)
    I = (Index{Lower}(1),Index{Interior}(2));
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[I] == (:apply,v,I)

    struct ScalingOperator{T,D} <: TensorMapping{T,D,D}
        λ::T
        size::NTuple{D,Int}
    end

    LazyTensors.apply(m::ScalingOperator{T,D}, v, I::Vararg{Index,D}) where {T,D} = m.λ*v[I]
    LazyTensors.range_size(m::ScalingOperator) = m.size
    LazyTensors.domain_size(m::ScalingOperator) = m.size

    m = ScalingOperator{Int,1}(2,(3,))
    v = [1,2,3]
    @test m*v isa AbstractVector
    @test m*v == [2,4,6]

    m = ScalingOperator{Int,2}(2,(2,2))
    v = [[1 2];[3 4]]
    @test m*v == [[2 4];[6 8]]
    I = (Index{Upper}(2),Index{Lower}(1))
    @test (m*v)[I] == 6
end

@testset "TensorMapping binary operations" begin
    struct ScalarMapping{T,R,D} <: TensorMapping{T,R,D}
        λ::T
        range_size::NTuple{R,Int}
        domain_size::NTuple{D,Int}
    end

    LazyTensors.apply(m::ScalarMapping{T,R,D}, v, I::Vararg{Index{<:Region}}) where {T,R,D} = m.λ*v[I...]
    LazyTensors.range_size(m::ScalarMapping) = m.domain_size
    LazyTensors.domain_size(m::ScalarMapping) = m.range_size

    A = ScalarMapping{Float64,1,1}(2.0, (3,), (3,))
    B = ScalarMapping{Float64,1,1}(3.0, (3,), (3,))

    v = [1.1,1.2,1.3]
    for i ∈ eachindex(v)
        @test ((A+B)*v)[i] == 2*v[i] + 3*v[i]
    end

    for i ∈ eachindex(v)
        @test ((A-B)*v)[i] == 2*v[i] - 3*v[i]
    end

    @test range_size(A+B) == range_size(A) == range_size(B)
    @test domain_size(A+B) == domain_size(A) == domain_size(B)
end

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

@testset "TensorMappingComposition" begin
    A = rand(2,3)
    B = rand(3,4)

    Ã = LazyLinearMap(A, (1,), (2,))
    B̃ = LazyLinearMap(B, (1,), (2,))

    @test Ã∘B̃ isa TensorMappingComposition
    @test range_size(Ã∘B̃) == (2,)
    @test domain_size(Ã∘B̃) == (4,)
    @test_throws SizeMismatch B̃∘Ã

    # @test @inbounds B̃∘Ã # Should not error even though dimensions don't match. (Since ]test runs with forced boundschecking this is currently not testable 2020-10-16)

    v = rand(4)
    @test Ã∘B̃*v ≈ A*B*v rtol=1e-14

    v = rand(2)
    @test (Ã∘B̃)'*v ≈ B'*A'*v rtol=1e-14
end

@testset "LazyLinearMap" begin
    # Test a standard matrix-vector product
    # mapping vectors of size 4 to vectors of size 3.
    A = rand(3,4)
    Ã = LazyLinearMap(A, (1,), (2,))
    v = rand(4)
    w = rand(3)

    @test Ã isa LazyLinearMap{T,1,1} where T
    @test Ã isa TensorMapping{T,1,1} where T
    @test range_size(Ã) == (3,)
    @test domain_size(Ã) == (4,)

    @test Ã*ones(4) ≈ A*ones(4) atol=5e-13
    @test Ã*v ≈ A*v atol=5e-13
    @test Ã'*w ≈ A'*w

    A = rand(2,3,4)
    @test_throws DomainError LazyLinearMap(A, (3,1), (2,))

    # Test more exotic mappings
    B = rand(3,4,2)
    # Map vectors of size 2 to matrices of size (3,4)
    B̃ = LazyLinearMap(B, (1,2), (3,))
    v = rand(2)

    @test range_size(B̃) == (3,4)
    @test domain_size(B̃) == (2,)
    @test B̃ isa TensorMapping{T,2,1} where T
    @test B̃*ones(2) ≈ B[:,:,1] + B[:,:,2] atol=5e-13
    @test B̃*v ≈ B[:,:,1]*v[1] + B[:,:,2]*v[2] atol=5e-13

    # Map matrices of size (3,2) to vectors of size 4
    B̃ = LazyLinearMap(B, (2,), (1,3))
    v = rand(3,2)

    @test range_size(B̃) == (4,)
    @test domain_size(B̃) == (3,2)
    @test B̃ isa TensorMapping{T,1,2} where T
    @test B̃*ones(3,2) ≈ B[1,:,1] + B[2,:,1] + B[3,:,1] +
                        B[1,:,2] + B[2,:,2] + B[3,:,2] atol=5e-13
    @test B̃*v ≈ B[1,:,1]*v[1,1] + B[2,:,1]*v[2,1] + B[3,:,1]*v[3,1] +
                B[1,:,2]v[1,2] + B[2,:,2]*v[2,2] + B[3,:,2]*v[3,2] atol=5e-13


    # TODO:
    # @inferred (B̃*v)[2]
end


@testset "IdentityMapping" begin
    @test IdentityMapping{Float64}((4,5)) isa IdentityMapping{T,2} where T
    @test IdentityMapping{Float64}((4,5)) isa TensorMapping{T,2,2} where T
    @test IdentityMapping{Float64}((4,5)) == IdentityMapping{Float64}(4,5)

    @test IdentityMapping(3,2) isa IdentityMapping{Float64,2}

    for sz ∈ [(4,5),(3,),(5,6,4)]
        I = IdentityMapping{Float64}(sz)
        v = rand(sz...)
        @test I*v == v
        @test I'*v == v

        @test range_size(I) == sz
        @test domain_size(I) == sz
    end

    I = IdentityMapping{Float64}((4,5))
    v = rand(4,5)
    @inferred (I*v)[3,2]
    @inferred (I'*v)[3,2]
    @inferred range_size(I)

    @inferred range_dim(I)
    @inferred domain_dim(I)

    Ã = rand(4,2)
    A = LazyLinearMap(Ã,(1,),(2,))
    I1 = IdentityMapping{Float64}(2)
    I2 = IdentityMapping{Float64}(4)
    @test A∘I1 == A
    @test I2∘A == A
    @test I1∘I1 == I1
    @test_throws SizeMismatch I1∘A
    @test_throws SizeMismatch A∘I2
    @test_throws SizeMismatch I1∘I2
end

@testset "InflatedTensorMapping" begin
    I(sz...) = IdentityMapping(sz...)

    Ã = rand(4,2)
    B̃ = rand(4,2,3)
    C̃ = rand(4,2,3)

    A = LazyLinearMap(Ã,(1,),(2,))
    B = LazyLinearMap(B̃,(1,2),(3,))
    C = LazyLinearMap(C̃,(1,),(2,3))

    @test InflatedTensorMapping(I(3,2), A, I(4)) isa TensorMapping{Float64, 4, 4}
    @test InflatedTensorMapping(I(3,2), B, I(4)) isa TensorMapping{Float64, 5, 4}
    @test InflatedTensorMapping(I(3), C, I(2,3)) isa TensorMapping{Float64, 4, 5}
    @test InflatedTensorMapping(C, I(2,3)) isa TensorMapping{Float64, 3, 4}
    @test InflatedTensorMapping(I(3), C) isa TensorMapping{Float64, 2, 3}
    @test InflatedTensorMapping(I(3), I(2,3)) isa TensorMapping{Float64, 3, 3}

    @test range_size(InflatedTensorMapping(I(3,2), A, I(4))) == (3,2,4,4)
    @test domain_size(InflatedTensorMapping(I(3,2), A, I(4))) == (3,2,2,4)

    @test range_size(InflatedTensorMapping(I(3,2), B, I(4))) == (3,2,4,2,4)
    @test domain_size(InflatedTensorMapping(I(3,2), B, I(4))) == (3,2,3,4)

    @test range_size(InflatedTensorMapping(I(3), C, I(2,3))) == (3,4,2,3)
    @test domain_size(InflatedTensorMapping(I(3), C, I(2,3))) == (3,2,3,2,3)

    @inferred range_size(InflatedTensorMapping(I(3,2), A, I(4))) == (3,2,4,4)
    @inferred domain_size(InflatedTensorMapping(I(3,2), A, I(4))) == (3,2,2,4)

    # Test InflatedTensorMapping mapping w. before and after
    tm = InflatedTensorMapping(I(3,2), A, I(4))
    v = rand(domain_size(tm)...)
    @tullio IAIv[a,b,c,d] := Ã[c,i]*v[a,b,i,d]
    @test tm*v ≈ IAIv rtol=1e-14
    @inferred LazyTensors.split_index(tm,1,1,1,1)

    # Test InflatedTensorMapping mapping w. before
    tm = InflatedTensorMapping(I(3,2), A)
    v = rand(domain_size(tm)...)
    @tullio IAIv[a,b,c] := Ã[c,i]*v[a,b,i]
    @test tm*v ≈ IAIv rtol=1e-14
    @inferred LazyTensors.split_index(tm,1,1,1)

    # Test InflatedTensorMapping mapping w. after
    tm = InflatedTensorMapping(A,I(4))
    v = rand(domain_size(tm)...)
    @tullio IAIv[c,d] := Ã[c,i]*v[i,d]
    @test tm*v ≈ IAIv rtol=1e-14
    @inferred LazyTensors.split_index(tm,1,1)

    struct ScalingOperator{T,D} <: TensorMapping{T,D,D}
        λ::T
        size::NTuple{D,Int}
    end

    LazyTensors.apply(m::ScalingOperator{T,D}, v, I::Vararg{Index,D}) where {T,D} = m.λ*v[I]
    LazyTensors.range_size(m::ScalingOperator) = m.size
    LazyTensors.domain_size(m::ScalingOperator) = m.size

    tm = InflatedTensorMapping(I(2,3),ScalingOperator(2.0, (3,2)),I(3,4))
    v = rand(domain_size(tm)...)

    @inferred LazyTensors.split_index(tm,1,2,3,2,2,4)
    @inferred apply(tm,v,Index{Unknown}.((1,2,3,2,2,4))...)
    @inferred (tm*v)[1,2,3,2,2,4]

    @testset "InflatedTensorMapping of InflatedTensorMapping" begin
        A = ScalingOperator(2.0,(2,3))
        itm = InflatedTensorMapping(I(3,2), A, I(4))
        @test  InflatedTensorMapping(I(4), itm, I(2)) == InflatedTensorMapping(I(4,3,2), A, I(4,2))
        @test  InflatedTensorMapping(itm, I(2)) == InflatedTensorMapping(I(3,2), A, I(4,2))
        @test  InflatedTensorMapping(I(4), itm) == InflatedTensorMapping(I(4,3,2), A, I(4))

        @test InflatedTensorMapping(I(2), I(2), I(2)) isa InflatedTensorMapping # The constructor should always return its type.
    end

end

@testset "slice_tuple" begin
    @test LazyTensors.slice_tuple((1,2,3),Val(1), Val(3)) == (1,2,3)
    @test LazyTensors.slice_tuple((1,2,3,4,5,6),Val(2), Val(5)) == (2,3,4,5)
    @test LazyTensors.slice_tuple((1,2,3,4,5,6),Val(1), Val(3)) == (1,2,3)
    @test LazyTensors.slice_tuple((1,2,3,4,5,6),Val(4), Val(6)) == (4,5,6)
end

@testset "flatten_tuple" begin
    @test LazyTensors.flatten_tuple((1,)) == (1,)
    @test LazyTensors.flatten_tuple((1,2,3,4,5,6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple((1,2,(3,4),5,6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple((1,2,(3,(4,5)),6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple(((1,2),(3,4),(5,),6)) == (1,2,3,4,5,6)
end


@testset "LazyOuterProduct" begin
    struct ScalingOperator{T,D} <: TensorMapping{T,D,D}
        λ::T
        size::NTuple{D,Int}
    end

    LazyTensors.apply(m::ScalingOperator{T,D}, v, I::Vararg{Index,D}) where {T,D} = m.λ*v[I]
    LazyTensors.range_size(m::ScalingOperator) = m.size
    LazyTensors.domain_size(m::ScalingOperator) = m.size

    A = ScalingOperator(2.0, (5,))
    B = ScalingOperator(3.0, (3,))
    C = ScalingOperator(5.0, (3,2))

    AB = LazyOuterProduct(A,B)
    @test AB isa TensorMapping{T,2,2} where T
    @test range_size(AB) == (5,3)
    @test domain_size(AB) == (5,3)

    v = rand(range_size(AB)...)
    @test AB*v == 6*v

    ABC = LazyOuterProduct(A,B,C)

    @test ABC isa TensorMapping{T,4,4} where T
    @test range_size(ABC) == (5,3,3,2)
    @test domain_size(ABC) == (5,3,3,2)

    @test A⊗B == AB
    @test A⊗B⊗C == ABC

    A = rand(3,2)
    B = rand(2,4,3)

    v₁ = rand(2,4,3)
    v₂ = rand(4,3,2)

    Ã = LazyLinearMap(A,(1,),(2,))
    B̃ = LazyLinearMap(B,(1,),(2,3))

    ÃB̃ = LazyOuterProduct(Ã,B̃)
    @tullio ABv[i,k] := A[i,j]*B[k,l,m]*v₁[j,l,m]
    @test ÃB̃*v₁ ≈ ABv

    B̃Ã = LazyOuterProduct(B̃,Ã)
    @tullio BAv[k,i] := A[i,j]*B[k,l,m]*v₂[l,m,j]
    @test B̃Ã*v₂ ≈ BAv

    @testset "Indentity mapping arguments" begin
        @test LazyOuterProduct(IdentityMapping(3,2), IdentityMapping(1,2)) == IdentityMapping(3,2,1,2)

        Ã = LazyLinearMap(A,(1,),(2,))
        @test LazyOuterProduct(IdentityMapping(3,2), Ã) == InflatedTensorMapping(IdentityMapping(3,2),Ã)
        @test LazyOuterProduct(Ã, IdentityMapping(3,2)) == InflatedTensorMapping(Ã,IdentityMapping(3,2))

        I1 = IdentityMapping(3,2)
        I2 = IdentityMapping(4)
        @test I1⊗Ã⊗I2 == InflatedTensorMapping(I1, Ã, I2)
    end

end

end
