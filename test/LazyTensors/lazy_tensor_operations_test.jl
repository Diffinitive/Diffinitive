using Test
using Sbplib.LazyTensors
using Sbplib.RegionIndices

using Tullio

struct TransposableDummyMapping{T,R,D} <: LazyTensor{T,R,D} end

LazyTensors.apply(m::TransposableDummyMapping{T,R}, v, I::Vararg{Any,R}) where {T,R} = :apply
LazyTensors.apply_transpose(m::TransposableDummyMapping{T,R,D}, v, I::Vararg{Any,D}) where {T,R,D} = :apply_transpose

LazyTensors.range_size(m::TransposableDummyMapping) = :range_size
LazyTensors.domain_size(m::TransposableDummyMapping) = :domain_size


struct SizeDoublingMapping{T,R,D} <: LazyTensor{T,R,D}
    domain_size::NTuple{D,Int}
end

LazyTensors.apply(m::SizeDoublingMapping{T,R}, v, i::Vararg{Any,R}) where {T,R} = (:apply,v,i)
LazyTensors.range_size(m::SizeDoublingMapping) = 2 .* m.domain_size
LazyTensors.domain_size(m::SizeDoublingMapping) = m.domain_size



@testset "Mapping transpose" begin
    m = TransposableDummyMapping{Float64,2,3}()
    @test m' isa LazyTensor{Float64, 3,2}
    @test m'' == m
    @test apply(m',zeros(Float64,(0,0)), 0, 0, 0) == :apply_transpose
    @test apply(m'',zeros(Float64,(0,0,0)), 0, 0) == :apply
    @test apply_transpose(m', zeros(Float64,(0,0,0)), 0, 0) == :apply

    @test range_size(m') == :domain_size
    @test domain_size(m') == :range_size
end


@testset "TensorApplication" begin
    m = SizeDoublingMapping{Int, 1, 1}((3,))
    mm = SizeDoublingMapping{Int, 1, 1}((6,))
    v = [0,1,2]
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[1] == (:apply,v,(1,))
    @test (mm*m*v)[1] == (:apply,m*v,(1,))
    @test (mm*m*v)[3] == (:apply,m*v,(3,))
    @test (mm*m*v)[6] == (:apply,m*v,(6,))
    @test_throws MethodError m*m

    @test (m*v)[CartesianIndex(2)] == (:apply,v,(2,))
    @test (mm*m*v)[CartesianIndex(2)] == (:apply,m*v,(2,))

    m = SizeDoublingMapping{Float64, 2, 2}((3,3))
    mm = SizeDoublingMapping{Float64, 2, 2}((6,6))
    v = ones(3,3)
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[1,2] == (:apply,v,(1,2))

    @test (m*v)[CartesianIndex(2,3)] == (:apply,v,(2,3))
    @test (mm*m*v)[CartesianIndex(4,3)] == (:apply,m*v,(4,3))

    m = ScalingTensor(2,(3,))
    v = [1,2,3]
    @test m*v isa AbstractVector
    @test m*v == [2,4,6]

    m = ScalingTensor(2,(2,2))
    v = [[1 2];[3 4]]
    @test m*v == [[2 4];[6 8]]
    @test (m*v)[2,1] == 6

    @testset "Error on index out of bounds" begin
        m = SizeDoublingMapping{Int, 1, 1}((3,))
        v = [0,1,2]

        @test_throws BoundsError (m*v)[0]
        @test_throws BoundsError (m*v)[7]
    end

    @testset "Error on unmatched dimensions" begin
        v = [0,1,2]
        m = SizeDoublingMapping{Int, 2, 1}((3,))
        @test_throws MethodError m*ones(Int,2,2)
        @test_throws MethodError m*m*v
    end

    @testset "Error on unmatched sizes" begin
        @test_throws DomainSizeMismatch ScalingTensor(2,(2,))*ones(3)
        @test_throws DomainSizeMismatch ScalingTensor(2,(2,))*ScalingTensor(2,(3,))*ones(3)
    end


    @testset "Type calculation" begin
        m = ScalingTensor(2,(3,))
        v = [1.,2.,3.]
        @test m*v isa AbstractVector{Float64}
        @test m*v == [2.,4.,6.]
        @inferred m*v
        @inferred (m*v)[1]

        m = ScalingTensor(2,(2,2))
        v = [[1. 2.];[3. 4.]]
        @test m*v == [[2. 4.];[6. 8.]]
        @test (m*v)[2,1] == 6.
        @inferred m*v
        @inferred (m*v)[1]

        m = ScalingTensor(2. +2. *im,(3,))
        v = [1.,2.,3.]
        @test m*v isa AbstractVector{ComplexF64}
        @test m*v == [2. + 2. *im, 4. + 4. *im, 6. + 6. *im]
        @inferred m*v
        @inferred (m*v)[1]

        m = ScalingTensor(1,(3,))
        v = [2. + 2. *im, 4. + 4. *im, 6. + 6. *im]
        @test m*v isa AbstractVector{ComplexF64}
        @test m*v == [2. + 2. *im, 4. + 4. *im, 6. + 6. *im]
        @inferred m*v
        @inferred (m*v)[1]

        m = ScalingTensor(2., (3,))
        v = [[1,2,3], [3,2,1],[1,3,1]]
        @test m*v isa AbstractVector{Vector{Float64}}
        @test m*v == [[2.,4.,6.], [6.,4.,2.],[2.,6.,2.]]
        @inferred m*v
        @inferred (m*v)[1]
    end
end


@testset "LazyTensor binary operations" begin
    A = ScalingTensor(2.0, (3,))
    B = ScalingTensor(3.0, (3,))

    v = [1.1,1.2,1.3]
    for i ∈ eachindex(v)
        @test ((A+B)*v)[i] == 2*v[i] + 3*v[i]
    end

    for i ∈ eachindex(v)
        @test ((A-B)*v)[i] == 2*v[i] - 3*v[i]
    end


    @test range_size(A+B) == range_size(A) == range_size(B)
    @test domain_size(A+B) == domain_size(A) == domain_size(B)

    @test ((A+B)*ComplexF64[1.1,1.2,1.3])[3] isa ComplexF64

    @testset "Error on unmatched sizes" begin
        @test_throws Union{DomainSizeMismatch, RangeSizeMismatch} ScalingTensor(2.0, (3,)) + ScalingTensor(2.0, (4,))

        @test_throws DomainSizeMismatch ScalingTensor(2.0, (4,)) + SizeDoublingMapping{Float64,1,1}((2,))
        @test_throws DomainSizeMismatch SizeDoublingMapping{Float64,1,1}((2,)) + ScalingTensor(2.0, (4,))
        @test_throws RangeSizeMismatch ScalingTensor(2.0, (2,)) + SizeDoublingMapping{Float64,1,1}((2,))
        @test_throws RangeSizeMismatch SizeDoublingMapping{Float64,1,1}((2,)) + ScalingTensor(2.0, (2,))
    end
end


@testset "TensorComposition" begin
    A = rand(2,3)
    B = rand(3,4)

    Ã = DenseTensor(A, (1,), (2,))
    B̃ = DenseTensor(B, (1,), (2,))

    @test Ã∘B̃ isa TensorComposition
    @test range_size(Ã∘B̃) == (2,)
    @test domain_size(Ã∘B̃) == (4,)
    @test_throws DomainSizeMismatch B̃∘Ã

    # @test @inbounds B̃∘Ã # Should not error even though dimensions don't match. (Since ]test runs with forced boundschecking this is currently not testable 2020-10-16)

    v = rand(4)
    @test Ã∘B̃*v ≈ A*B*v rtol=1e-14

    v = rand(2)
    @test (Ã∘B̃)'*v ≈ B'*A'*v rtol=1e-14

    @test (Ã∘B̃*ComplexF64[1.,2.,3.,4.])[1] isa ComplexF64
    @test ((Ã∘B̃)'*ComplexF64[1.,2.])[1] isa ComplexF64

    a = 2.
    v = rand(3)
    @test a*Ã isa TensorComposition
    @test a*Ã == Ã*a
    @test range_size(a*Ã) == range_size(Ã)
    @test domain_size(a*Ã) == domain_size(Ã)
    @test a*Ã*v ≈ a.*A*v rtol=1e-14
end


@testset "InflatedTensor" begin
    I(sz...) = IdentityTensor(sz...)

    Ã = rand(4,2)
    B̃ = rand(4,2,3)
    C̃ = rand(4,2,3)

    A = DenseTensor(Ã,(1,),(2,))
    B = DenseTensor(B̃,(1,2),(3,))
    C = DenseTensor(C̃,(1,),(2,3))

    @testset "Constructors" begin
        @test InflatedTensor(I(3,2), A, I(4)) isa LazyTensor{Float64, 4, 4}
        @test InflatedTensor(I(3,2), B, I(4)) isa LazyTensor{Float64, 5, 4}
        @test InflatedTensor(I(3), C, I(2,3)) isa LazyTensor{Float64, 4, 5}
        @test InflatedTensor(C, I(2,3)) isa LazyTensor{Float64, 3, 4}
        @test InflatedTensor(I(3), C) isa LazyTensor{Float64, 2, 3}
        @test InflatedTensor(I(3), I(2,3)) isa LazyTensor{Float64, 3, 3}
    end

    @testset "Range and domain size" begin
        @test range_size(InflatedTensor(I(3,2), A, I(4))) == (3,2,4,4)
        @test domain_size(InflatedTensor(I(3,2), A, I(4))) == (3,2,2,4)

        @test range_size(InflatedTensor(I(3,2), B, I(4))) == (3,2,4,2,4)
        @test domain_size(InflatedTensor(I(3,2), B, I(4))) == (3,2,3,4)

        @test range_size(InflatedTensor(I(3), C, I(2,3))) == (3,4,2,3)
        @test domain_size(InflatedTensor(I(3), C, I(2,3))) == (3,2,3,2,3)

        @inferred range_size(InflatedTensor(I(3,2), A, I(4))) == (3,2,4,4)
        @inferred domain_size(InflatedTensor(I(3,2), A, I(4))) == (3,2,2,4)
    end

    @testset "Application" begin
        # Testing regular application and transposed application with inflation "before", "after" and "before and after".
        # The inflated tensor mappings are chosen to preserve, reduce and increase the dimension of the result compared to the input.
        cases = [
            (
                InflatedTensor(I(3,2), A, I(4)),
                (v-> @tullio res[a,b,c,d] := Ã[c,i]*v[a,b,i,d]), # Expected result of apply
                (v-> @tullio res[a,b,c,d] := Ã[i,c]*v[a,b,i,d]), # Expected result of apply_transpose
            ),
            (
                InflatedTensor(I(3,2), B, I(4)),
                (v-> @tullio res[a,b,c,d,e] := B̃[c,d,i]*v[a,b,i,e]),
                (v-> @tullio res[a,b,c,d] := B̃[i,j,c]*v[a,b,i,j,d]),
            ),
            (
                InflatedTensor(I(3,2), C, I(4)),
                (v-> @tullio res[a,b,c,d] := C̃[c,i,j]*v[a,b,i,j,d]),
                (v-> @tullio res[a,b,c,d,e] := C̃[i,c,d]*v[a,b,i,e]),
            ),
            (
                InflatedTensor(I(3,2), A),
                (v-> @tullio res[a,b,c] := Ã[c,i]*v[a,b,i]),
                (v-> @tullio res[a,b,c] := Ã[i,c]*v[a,b,i]),
            ),
            (
                InflatedTensor(I(3,2), B),
                (v-> @tullio res[a,b,c,d] := B̃[c,d,i]*v[a,b,i]),
                (v-> @tullio res[a,b,c] := B̃[i,j,c]*v[a,b,i,j]),
            ),
            (
                InflatedTensor(I(3,2), C),
                (v-> @tullio res[a,b,c] := C̃[c,i,j]*v[a,b,i,j]),
                (v-> @tullio res[a,b,c,d] := C̃[i,c,d]*v[a,b,i]),
            ),
            (
                InflatedTensor(A,I(4)),
                (v-> @tullio res[a,b] := Ã[a,i]*v[i,b]),
                (v-> @tullio res[a,b] := Ã[i,a]*v[i,b]),
            ),
            (
                InflatedTensor(B,I(4)),
                (v-> @tullio res[a,b,c] := B̃[a,b,i]*v[i,c]),
                (v-> @tullio res[a,b] := B̃[i,j,a]*v[i,j,b]),
            ),
            (
                InflatedTensor(C,I(4)),
                (v-> @tullio res[a,b] := C̃[a,i,j]*v[i,j,b]),
                (v-> @tullio res[a,b,c] := C̃[i,a,b]*v[i,c]),
            ),
        ]

        @testset "$tm" for (tm, true_apply, true_apply_transpose) ∈ cases
            v = rand(domain_size(tm)...)
            @test tm*v ≈ true_apply(v) rtol=1e-14

            v = rand(range_size(tm)...)
            @test tm'*v ≈ true_apply_transpose(v) rtol=1e-14
        end

        @testset "application to other type" begin
            tm = InflatedTensor(I(3,2), A, I(4))

            v = rand(ComplexF64, domain_size(tm)...)
            @test (tm*v)[1,2,3,1] isa ComplexF64

            v = rand(ComplexF64, domain_size(tm')...)
            @test (tm'*v)[1,2,2,1] isa ComplexF64
        end

        @testset "Inference of application" begin
            tm = InflatedTensor(I(2,3),ScalingTensor(2.0, (3,2)),I(3,4))
            v = rand(domain_size(tm)...)

            @inferred apply(tm,v,1,2,3,2,2,4)
            @inferred (tm*v)[1,2,3,2,2,4]
        end
    end

    @testset "InflatedTensor of InflatedTensor" begin
        A = ScalingTensor(2.0,(2,3))
        itm = InflatedTensor(I(3,2), A, I(4))
        @test  InflatedTensor(I(4), itm, I(2)) == InflatedTensor(I(4,3,2), A, I(4,2))
        @test  InflatedTensor(itm, I(2)) == InflatedTensor(I(3,2), A, I(4,2))
        @test  InflatedTensor(I(4), itm) == InflatedTensor(I(4,3,2), A, I(4))

        @test InflatedTensor(I(2), I(2), I(2)) isa InflatedTensor # The constructor should always return its type.
    end
end

@testset "LazyOuterProduct" begin
    A = ScalingTensor(2.0, (5,))
    B = ScalingTensor(3.0, (3,))
    C = ScalingTensor(5.0, (3,2))

    AB = LazyOuterProduct(A,B)
    @test AB isa LazyTensor{T,2,2} where T
    @test range_size(AB) == (5,3)
    @test domain_size(AB) == (5,3)

    v = rand(range_size(AB)...)
    @test AB*v == 6*v

    ABC = LazyOuterProduct(A,B,C)

    @test ABC isa LazyTensor{T,4,4} where T
    @test range_size(ABC) == (5,3,3,2)
    @test domain_size(ABC) == (5,3,3,2)

    @test A⊗B == AB
    @test A⊗B⊗C == ABC

    A = rand(3,2)
    B = rand(2,4,3)

    v₁ = rand(2,4,3)
    v₂ = rand(4,3,2)

    Ã = DenseTensor(A,(1,),(2,))
    B̃ = DenseTensor(B,(1,),(2,3))

    ÃB̃ = LazyOuterProduct(Ã,B̃)
    @tullio ABv[i,k] := A[i,j]*B[k,l,m]*v₁[j,l,m]
    @test ÃB̃*v₁ ≈ ABv

    B̃Ã = LazyOuterProduct(B̃,Ã)
    @tullio BAv[k,i] := A[i,j]*B[k,l,m]*v₂[l,m,j]
    @test B̃Ã*v₂ ≈ BAv

    @testset "Indentity mapping arguments" begin
        @test LazyOuterProduct(IdentityTensor(3,2), IdentityTensor(1,2)) == IdentityTensor(3,2,1,2)

        Ã = DenseTensor(A,(1,),(2,))
        @test LazyOuterProduct(IdentityTensor(3,2), Ã) == InflatedTensor(IdentityTensor(3,2),Ã)
        @test LazyOuterProduct(Ã, IdentityTensor(3,2)) == InflatedTensor(Ã,IdentityTensor(3,2))

        I1 = IdentityTensor(3,2)
        I2 = IdentityTensor(4)
        @test I1⊗Ã⊗I2 == InflatedTensor(I1, Ã, I2)
    end
end

@testset "inflate" begin
    I = LazyTensors.inflate(IdentityTensor(),(3,4,5,6), 2)
    @test I isa LazyTensor{Float64, 3,3}
    @test range_size(I) == (3,5,6)
    @test domain_size(I) == (3,5,6)

    @test LazyTensors.inflate(ScalingTensor(1., (4,)),(3,4,5,6), 1) == InflatedTensor(IdentityTensor{Float64}(),ScalingTensor(1., (4,)),IdentityTensor(4,5,6))
    @test LazyTensors.inflate(ScalingTensor(2., (1,)),(3,4,5,6), 2) == InflatedTensor(IdentityTensor(3),ScalingTensor(2., (1,)),IdentityTensor(5,6))
    @test LazyTensors.inflate(ScalingTensor(3., (6,)),(3,4,5,6), 4) == InflatedTensor(IdentityTensor(3,4,5),ScalingTensor(3., (6,)),IdentityTensor{Float64}())

    @test_throws BoundsError LazyTensors.inflate(ScalingTensor(1., (4,)),(3,4,5,6), 0)
    @test_throws BoundsError LazyTensors.inflate(ScalingTensor(1., (4,)),(3,4,5,6), 5)
end
