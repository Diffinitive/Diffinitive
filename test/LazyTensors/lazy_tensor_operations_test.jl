using Test
using Sbplib.LazyTensors
using Sbplib.RegionIndices

using Tullio

@testset "Mapping transpose" begin
    struct DummyMapping{T,R,D} <: LazyTensor{T,R,D} end

    LazyTensors.apply(m::DummyMapping{T,R}, v, I::Vararg{Any,R}) where {T,R} = :apply
    LazyTensors.apply_transpose(m::DummyMapping{T,R,D}, v, I::Vararg{Any,D}) where {T,R,D} = :apply_transpose

    LazyTensors.range_size(m::DummyMapping) = :range_size
    LazyTensors.domain_size(m::DummyMapping) = :domain_size

    m = DummyMapping{Float64,2,3}()
    @test m' isa LazyTensor{Float64, 3,2}
    @test m'' == m
    @test apply(m',zeros(Float64,(0,0)), 0, 0, 0) == :apply_transpose
    @test apply(m'',zeros(Float64,(0,0,0)), 0, 0) == :apply
    @test apply_transpose(m', zeros(Float64,(0,0,0)), 0, 0) == :apply

    @test range_size(m') == :domain_size
    @test domain_size(m') == :range_size
end

@testset "TensorApplication" begin
    struct SizeDoublingMapping{T,R,D} <: LazyTensor{T,R,D}
        domain_size::NTuple{D,Int}
    end

    LazyTensors.apply(m::SizeDoublingMapping{T,R}, v, i::Vararg{Any,R}) where {T,R} = (:apply,v,i)
    LazyTensors.range_size(m::SizeDoublingMapping) = 2 .* m.domain_size
    LazyTensors.domain_size(m::SizeDoublingMapping) = m.domain_size


    m = SizeDoublingMapping{Int, 1, 1}((3,))
    v = [0,1,2]
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[0] == (:apply,v,(0,))
    @test (m*m*v)[1] == (:apply,m*v,(1,))
    @test (m*m*v)[3] == (:apply,m*v,(3,))
    @test (m*m*v)[6] == (:apply,m*v,(6,))
    @test_broken BoundsError == (m*m*v)[0]
    @test_broken BoundsError == (m*m*v)[7]
    @test_throws MethodError m*m

    @test (m*v)[CartesianIndex(2)] == (:apply,v,(2,))
    @test (m*m*v)[CartesianIndex(2)] == (:apply,m*v,(2,))

    m = SizeDoublingMapping{Int, 2, 1}((3,))
    @test_throws MethodError m*ones(Int,2,2)
    @test_throws MethodError m*m*v

    m = SizeDoublingMapping{Float64, 2, 2}((3,3))
    v = ones(3,3)
    @test size(m*v) == 2 .*size(v)
    @test (m*v)[1,2] == (:apply,v,(1,2))

    @test (m*v)[CartesianIndex(2,3)] == (:apply,v,(2,3))
    @test (m*m*v)[CartesianIndex(4,3)] == (:apply,m*v,(4,3))

    m = ScalingTensor(2,(3,))
    v = [1,2,3]
    @test m*v isa AbstractVector
    @test m*v == [2,4,6]

    m = ScalingTensor(2,(2,2))
    v = [[1 2];[3 4]]
    @test m*v == [[2 4];[6 8]]
    @test (m*v)[2,1] == 6

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

    # TODO: Test with size changing tm
    # TODO: Test for mismatch in dimensions (SizeMismatch?)

    @test range_size(A+B) == range_size(A) == range_size(B)
    @test domain_size(A+B) == domain_size(A) == domain_size(B)

    @test ((A+B)*ComplexF64[1.1,1.2,1.3])[3] isa ComplexF64
end


@testset "LazyTensorComposition" begin
    A = rand(2,3)
    B = rand(3,4)

    Ã = LazyLinearMap(A, (1,), (2,))
    B̃ = LazyLinearMap(B, (1,), (2,))

    @test Ã∘B̃ isa LazyTensorComposition
    @test range_size(Ã∘B̃) == (2,)
    @test domain_size(Ã∘B̃) == (4,)
    @test_throws SizeMismatch B̃∘Ã

    # @test @inbounds B̃∘Ã # Should not error even though dimensions don't match. (Since ]test runs with forced boundschecking this is currently not testable 2020-10-16)

    v = rand(4)
    @test Ã∘B̃*v ≈ A*B*v rtol=1e-14

    v = rand(2)
    @test (Ã∘B̃)'*v ≈ B'*A'*v rtol=1e-14

    @test (Ã∘B̃*ComplexF64[1.,2.,3.,4.])[1] isa ComplexF64
    @test ((Ã∘B̃)'*ComplexF64[1.,2.])[1] isa ComplexF64
end

@testset "LazyLinearMap" begin
    # Test a standard matrix-vector product
    # mapping vectors of size 4 to vectors of size 3.
    A = rand(3,4)
    Ã = LazyLinearMap(A, (1,), (2,))
    v = rand(4)
    w = rand(3)

    @test Ã isa LazyLinearMap{T,1,1} where T
    @test Ã isa LazyTensor{T,1,1} where T
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
    @test B̃ isa LazyTensor{T,2,1} where T
    @test B̃*ones(2) ≈ B[:,:,1] + B[:,:,2] atol=5e-13
    @test B̃*v ≈ B[:,:,1]*v[1] + B[:,:,2]*v[2] atol=5e-13

    # Map matrices of size (3,2) to vectors of size 4
    B̃ = LazyLinearMap(B, (2,), (1,3))
    v = rand(3,2)

    @test range_size(B̃) == (4,)
    @test domain_size(B̃) == (3,2)
    @test B̃ isa LazyTensor{T,1,2} where T
    @test B̃*ones(3,2) ≈ B[1,:,1] + B[2,:,1] + B[3,:,1] +
                        B[1,:,2] + B[2,:,2] + B[3,:,2] atol=5e-13
    @test B̃*v ≈ B[1,:,1]*v[1,1] + B[2,:,1]*v[2,1] + B[3,:,1]*v[3,1] +
                B[1,:,2]v[1,2] + B[2,:,2]*v[2,2] + B[3,:,2]*v[3,2] atol=5e-13


    # TODO:
    # @inferred (B̃*v)[2]
end


@testset "IdentityTensor" begin
    @test IdentityTensor{Float64}((4,5)) isa IdentityTensor{T,2} where T
    @test IdentityTensor{Float64}((4,5)) isa LazyTensor{T,2,2} where T
    @test IdentityTensor{Float64}((4,5)) == IdentityTensor{Float64}(4,5)

    @test IdentityTensor(3,2) isa IdentityTensor{Float64,2}

    for sz ∈ [(4,5),(3,),(5,6,4)]
        I = IdentityTensor{Float64}(sz)
        v = rand(sz...)
        @test I*v == v
        @test I'*v == v

        v = rand(ComplexF64,sz...)
        @test I*v == v
        @test I'*v == v

        @test range_size(I) == sz
        @test domain_size(I) == sz
    end

    I = IdentityTensor{Float64}((4,5))
    v = rand(4,5)
    @inferred (I*v)[3,2]
    @inferred (I'*v)[3,2]
    @inferred range_size(I)

    @inferred range_dim(I)
    @inferred domain_dim(I)

    Ã = rand(4,2)
    A = LazyLinearMap(Ã,(1,),(2,))
    I1 = IdentityTensor{Float64}(2)
    I2 = IdentityTensor{Float64}(4)
    @test A∘I1 == A
    @test I2∘A == A
    @test I1∘I1 == I1
    @test_throws SizeMismatch I1∘A
    @test_throws SizeMismatch A∘I2
    @test_throws SizeMismatch I1∘I2
end

@testset "ScalingTensor" begin
    st = ScalingTensor(2.,(3,4))
    @test st isa LazyTensor{Float64, 2, 2}
    @test range_size(st) == (3,4)
    @test domain_size(st) == (3,4)

    v = rand(3,4)
    @test st*v == 2.0 .* v
    @test st'*v == 2.0 .* v

    @inferred (st*v)[2,2]
    @inferred (st'*v)[2,2]
end

@testset "InflatedLazyTensor" begin
    I(sz...) = IdentityTensor(sz...)

    Ã = rand(4,2)
    B̃ = rand(4,2,3)
    C̃ = rand(4,2,3)

    A = LazyLinearMap(Ã,(1,),(2,))
    B = LazyLinearMap(B̃,(1,2),(3,))
    C = LazyLinearMap(C̃,(1,),(2,3))

    @testset "Constructors" begin
        @test InflatedLazyTensor(I(3,2), A, I(4)) isa LazyTensor{Float64, 4, 4}
        @test InflatedLazyTensor(I(3,2), B, I(4)) isa LazyTensor{Float64, 5, 4}
        @test InflatedLazyTensor(I(3), C, I(2,3)) isa LazyTensor{Float64, 4, 5}
        @test InflatedLazyTensor(C, I(2,3)) isa LazyTensor{Float64, 3, 4}
        @test InflatedLazyTensor(I(3), C) isa LazyTensor{Float64, 2, 3}
        @test InflatedLazyTensor(I(3), I(2,3)) isa LazyTensor{Float64, 3, 3}
    end

    @testset "Range and domain size" begin
        @test range_size(InflatedLazyTensor(I(3,2), A, I(4))) == (3,2,4,4)
        @test domain_size(InflatedLazyTensor(I(3,2), A, I(4))) == (3,2,2,4)

        @test range_size(InflatedLazyTensor(I(3,2), B, I(4))) == (3,2,4,2,4)
        @test domain_size(InflatedLazyTensor(I(3,2), B, I(4))) == (3,2,3,4)

        @test range_size(InflatedLazyTensor(I(3), C, I(2,3))) == (3,4,2,3)
        @test domain_size(InflatedLazyTensor(I(3), C, I(2,3))) == (3,2,3,2,3)

        @inferred range_size(InflatedLazyTensor(I(3,2), A, I(4))) == (3,2,4,4)
        @inferred domain_size(InflatedLazyTensor(I(3,2), A, I(4))) == (3,2,2,4)
    end

    @testset "Application" begin
        # Testing regular application and transposed application with inflation "before", "after" and "before and after".
        # The inflated tensor mappings are chosen to preserve, reduce and increase the dimension of the result compared to the input.
        tests = [
            (
                InflatedLazyTensor(I(3,2), A, I(4)),
                (v-> @tullio res[a,b,c,d] := Ã[c,i]*v[a,b,i,d]), # Expected result of apply
                (v-> @tullio res[a,b,c,d] := Ã[i,c]*v[a,b,i,d]), # Expected result of apply_transpose
            ),
            (
                InflatedLazyTensor(I(3,2), B, I(4)),
                (v-> @tullio res[a,b,c,d,e] := B̃[c,d,i]*v[a,b,i,e]),
                (v-> @tullio res[a,b,c,d] := B̃[i,j,c]*v[a,b,i,j,d]),
            ),
            (
                InflatedLazyTensor(I(3,2), C, I(4)),
                (v-> @tullio res[a,b,c,d] := C̃[c,i,j]*v[a,b,i,j,d]),
                (v-> @tullio res[a,b,c,d,e] := C̃[i,c,d]*v[a,b,i,e]),
            ),
            (
                InflatedLazyTensor(I(3,2), A),
                (v-> @tullio res[a,b,c] := Ã[c,i]*v[a,b,i]),
                (v-> @tullio res[a,b,c] := Ã[i,c]*v[a,b,i]),
            ),
            (
                InflatedLazyTensor(I(3,2), B),
                (v-> @tullio res[a,b,c,d] := B̃[c,d,i]*v[a,b,i]),
                (v-> @tullio res[a,b,c] := B̃[i,j,c]*v[a,b,i,j]),
            ),
            (
                InflatedLazyTensor(I(3,2), C),
                (v-> @tullio res[a,b,c] := C̃[c,i,j]*v[a,b,i,j]),
                (v-> @tullio res[a,b,c,d] := C̃[i,c,d]*v[a,b,i]),
            ),
            (
                InflatedLazyTensor(A,I(4)),
                (v-> @tullio res[a,b] := Ã[a,i]*v[i,b]),
                (v-> @tullio res[a,b] := Ã[i,a]*v[i,b]),
            ),
            (
                InflatedLazyTensor(B,I(4)),
                (v-> @tullio res[a,b,c] := B̃[a,b,i]*v[i,c]),
                (v-> @tullio res[a,b] := B̃[i,j,a]*v[i,j,b]),
            ),
            (
                InflatedLazyTensor(C,I(4)),
                (v-> @tullio res[a,b] := C̃[a,i,j]*v[i,j,b]),
                (v-> @tullio res[a,b,c] := C̃[i,a,b]*v[i,c]),
            ),
        ]

        @testset "apply" begin
            for i ∈ 1:length(tests)
                tm = tests[i][1]
                v = rand(domain_size(tm)...)
                true_value = tests[i][2](v)
                @test tm*v ≈ true_value rtol=1e-14
            end
        end

        @testset "apply_transpose" begin
            for i ∈ 1:length(tests)
                tm = tests[i][1]
                v = rand(range_size(tm)...)
                true_value = tests[i][3](v)
                @test tm'*v ≈ true_value rtol=1e-14
            end
        end

        @testset "application to other type" begin
            tm = InflatedLazyTensor(I(3,2), A, I(4))

            v = rand(ComplexF64, domain_size(tm)...)
            @test (tm*v)[1,2,3,1] isa ComplexF64

            v = rand(ComplexF64, domain_size(tm')...)
            @test (tm'*v)[1,2,2,1] isa ComplexF64
        end

        @testset "Inference of application" begin
            tm = InflatedLazyTensor(I(2,3),ScalingTensor(2.0, (3,2)),I(3,4))
            v = rand(domain_size(tm)...)

            @inferred apply(tm,v,1,2,3,2,2,4)
            @inferred (tm*v)[1,2,3,2,2,4]
        end
    end

    @testset "InflatedLazyTensor of InflatedLazyTensor" begin
        A = ScalingTensor(2.0,(2,3))
        itm = InflatedLazyTensor(I(3,2), A, I(4))
        @test  InflatedLazyTensor(I(4), itm, I(2)) == InflatedLazyTensor(I(4,3,2), A, I(4,2))
        @test  InflatedLazyTensor(itm, I(2)) == InflatedLazyTensor(I(3,2), A, I(4,2))
        @test  InflatedLazyTensor(I(4), itm) == InflatedLazyTensor(I(4,3,2), A, I(4))

        @test InflatedLazyTensor(I(2), I(2), I(2)) isa InflatedLazyTensor # The constructor should always return its type.
    end
end

@testset "split_index" begin
    @test LazyTensors.split_index(Val(2),Val(1),Val(2),Val(2),1,2,3,4,5,6) == ((1,2,:,5,6),(3,4))
    @test LazyTensors.split_index(Val(2),Val(3),Val(2),Val(2),1,2,3,4,5,6) == ((1,2,:,:,:,5,6),(3,4))
    @test LazyTensors.split_index(Val(3),Val(1),Val(1),Val(2),1,2,3,4,5,6) == ((1,2,3,:,5,6),(4,))
    @test LazyTensors.split_index(Val(3),Val(2),Val(1),Val(2),1,2,3,4,5,6) == ((1,2,3,:,:,5,6),(4,))
    @test LazyTensors.split_index(Val(1),Val(1),Val(2),Val(3),1,2,3,4,5,6) == ((1,:,4,5,6),(2,3))
    @test LazyTensors.split_index(Val(1),Val(2),Val(2),Val(3),1,2,3,4,5,6) == ((1,:,:,4,5,6),(2,3))

    @test LazyTensors.split_index(Val(0),Val(1),Val(3),Val(3),1,2,3,4,5,6) == ((:,4,5,6),(1,2,3))
    @test LazyTensors.split_index(Val(3),Val(1),Val(3),Val(0),1,2,3,4,5,6) == ((1,2,3,:),(4,5,6))

    @inferred LazyTensors.split_index(Val(2),Val(3),Val(2),Val(2),1,2,3,2,2,4)
end

@testset "slice_tuple" begin
    @test LazyTensors.slice_tuple((1,2,3),Val(1), Val(3)) == (1,2,3)
    @test LazyTensors.slice_tuple((1,2,3,4,5,6),Val(2), Val(5)) == (2,3,4,5)
    @test LazyTensors.slice_tuple((1,2,3,4,5,6),Val(1), Val(3)) == (1,2,3)
    @test LazyTensors.slice_tuple((1,2,3,4,5,6),Val(4), Val(6)) == (4,5,6)
end

@testset "split_tuple" begin
    @testset "2 parts" begin
        @test LazyTensors.split_tuple((),Val(0)) == ((),())
        @test LazyTensors.split_tuple((1,),Val(0)) == ((),(1,))
        @test LazyTensors.split_tuple((1,),Val(1)) == ((1,),())

        @test LazyTensors.split_tuple((1,2,3,4),Val(0)) == ((),(1,2,3,4))
        @test LazyTensors.split_tuple((1,2,3,4),Val(1)) == ((1,),(2,3,4))
        @test LazyTensors.split_tuple((1,2,3,4),Val(2)) == ((1,2),(3,4))
        @test LazyTensors.split_tuple((1,2,3,4),Val(3)) == ((1,2,3),(4,))
        @test LazyTensors.split_tuple((1,2,3,4),Val(4)) == ((1,2,3,4),())

        @test LazyTensors.split_tuple((1,2,true,4),Val(3)) == ((1,2,true),(4,))

        @inferred LazyTensors.split_tuple((1,2,3,4),Val(3))
        @inferred LazyTensors.split_tuple((1,2,true,4),Val(3))
    end

    @testset "3 parts" begin
        @test LazyTensors.split_tuple((),Val(0),Val(0)) == ((),(),())
        @test LazyTensors.split_tuple((1,2,3),Val(1), Val(1)) == ((1,),(2,),(3,))
        @test LazyTensors.split_tuple((1,true,3),Val(1), Val(1)) == ((1,),(true,),(3,))

        @test LazyTensors.split_tuple((1,2,3,4,5,6),Val(1),Val(2)) == ((1,),(2,3),(4,5,6))
        @test LazyTensors.split_tuple((1,2,3,4,5,6),Val(3),Val(2)) == ((1,2,3),(4,5),(6,))

        @inferred LazyTensors.split_tuple((1,2,3,4,5,6),Val(3),Val(2))
        @inferred LazyTensors.split_tuple((1,true,3),Val(1), Val(1))
    end
end

@testset "flatten_tuple" begin
    @test LazyTensors.flatten_tuple((1,)) == (1,)
    @test LazyTensors.flatten_tuple((1,2,3,4,5,6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple((1,2,(3,4),5,6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple((1,2,(3,(4,5)),6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple(((1,2),(3,4),(5,),6)) == (1,2,3,4,5,6)
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

    Ã = LazyLinearMap(A,(1,),(2,))
    B̃ = LazyLinearMap(B,(1,),(2,3))

    ÃB̃ = LazyOuterProduct(Ã,B̃)
    @tullio ABv[i,k] := A[i,j]*B[k,l,m]*v₁[j,l,m]
    @test ÃB̃*v₁ ≈ ABv

    B̃Ã = LazyOuterProduct(B̃,Ã)
    @tullio BAv[k,i] := A[i,j]*B[k,l,m]*v₂[l,m,j]
    @test B̃Ã*v₂ ≈ BAv

    @testset "Indentity mapping arguments" begin
        @test LazyOuterProduct(IdentityTensor(3,2), IdentityTensor(1,2)) == IdentityTensor(3,2,1,2)

        Ã = LazyLinearMap(A,(1,),(2,))
        @test LazyOuterProduct(IdentityTensor(3,2), Ã) == InflatedLazyTensor(IdentityTensor(3,2),Ã)
        @test LazyOuterProduct(Ã, IdentityTensor(3,2)) == InflatedLazyTensor(Ã,IdentityTensor(3,2))

        I1 = IdentityTensor(3,2)
        I2 = IdentityTensor(4)
        @test I1⊗Ã⊗I2 == InflatedLazyTensor(I1, Ã, I2)
    end

end
