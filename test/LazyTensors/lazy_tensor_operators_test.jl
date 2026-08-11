using Test
using Diffinitive.LazyTensors
using StaticArrays

struct OperatorSizeDoublingMapping{R,D} <: LazyTensor{R,D}
    domain_size::NTuple{D,Int}
end

LazyTensors.apply(m::OperatorSizeDoublingMapping{R}, v, i::Vararg{Any,R}) where {R} = (:apply, v, i)
LazyTensors.range_size(m::OperatorSizeDoublingMapping) = 2 .* m.domain_size
LazyTensors.domain_size(m::OperatorSizeDoublingMapping) = m.domain_size

@testset "application operator" begin
    m = OperatorSizeDoublingMapping{1, 1}((3,))
    mm = OperatorSizeDoublingMapping{1, 1}((6,))
    v = [0, 1, 2]

    @test m * v == TensorApplication(m, v)
    @test mm * m * v == TensorApplication(mm, TensorApplication(m, v))

    @test_throws MethodError m * m

    @test IdentityTensor(3) * v === v
end

@testset "scalar multiplication operator" begin
    A = DenseTensor(rand(2, 3), (1,), (2,))
    a = 2.0

    @test a * A == TensorComposition(ScalingTensor(a, range_size(A)), A)
    @test A * a == TensorComposition(ScalingTensor(a, range_size(A)), A)

    S = rand(SMatrix{3,3})
    @test S*A == TensorComposition(ScalingTensor(S, range_size(A)), A)
end

@testset "addition and subtraction operators" begin
    A = ScalingTensor(1.0, (3,))
    B = ScalingTensor(2.0, (3,))
    C = ScalingTensor(3.0, (3,))
    D = ScalingTensor(4.0, (3,))

    @test A + B == TensorSum(A, B)
    @test A + B + C + D == TensorSum(A, B, C, D)
    @test A + B - C + D == TensorSum(A, B, TensorNegation(C), D)

    @test TensorSum(A, B) + TensorSum(C, D) == TensorSum(A, B, C, D)
    @test TensorSum(A, B) + C == TensorSum(A, B, C)
    @test A + TensorSum(B, C) == TensorSum(A, B, C)
    @test TensorSum(A, B) + C + TensorSum(D,A) == TensorSum(A, B, C, D, A)

    @test -A == TensorNegation(A)
    @test -A - B - C - D == TensorSum(TensorNegation(A), TensorNegation(B), TensorNegation(C), TensorNegation(D))

    @testset "ZeroTensor arguments" begin
        A = ScalingTensor(1.0, (3, 3))
        AB = TensorSum(A, ScalingTensor(2.0, (3, 3)))

        @test ZeroTensor(3, 3) + A == A + ZeroTensor(3, 3) == A
        @test ZeroTensor(3, 3) + ZeroTensor(3, 3) == ZeroTensor(3, 3)
        @test AB + ZeroTensor(3, 3) == ZeroTensor(3, 3) + AB == AB
        @test -ZeroTensor(3, 3) == ZeroTensor(3, 3)
        @test_throws DomainSizeMismatch ZeroTensor((3, 3), (2, 3)) + A
        @test_throws RangeSizeMismatch ZeroTensor((1, 3), (3, 3)) + A
    end
end

@testset "composition operator" begin
    A = rand(2, 3)
    B = rand(3, 4)
    C = rand(4, 5)

    Ã = DenseTensor(A, (1,), (2,))
    B̃ = DenseTensor(B, (1,), (2,))
    C̃ = DenseTensor(C, (1,), (2,))

    @test Ã ∘ B̃ == TensorComposition(Ã, B̃)
    @test_throws DomainSizeMismatch B̃ ∘ Ã

    @test (Ã ∘ B̃) ∘ C̃ == TensorComposition(Ã, TensorComposition(B̃, C̃))

    @testset "IdentityTensor arguments" begin
        @test IdentityTensor(range_size(Ã)) ∘ Ã == Ã ∘ IdentityTensor(domain_size(Ã)) == Ã
        @test IdentityTensor(range_size(Ã)) ∘ IdentityTensor(range_size(Ã)) == IdentityTensor(range_size(Ã))
        @test (Ã ∘ B̃) ∘ IdentityTensor(domain_size(B̃)) == Ã ∘ B̃
        @test IdentityTensor(range_size(Ã))∘(Ã ∘ B̃) == Ã ∘ B̃
        @test_throws DomainSizeMismatch Ã ∘ IdentityTensor(range_size(Ã))
    end

    @testset "ZeroTensor arguments" begin
        @test ZeroTensor((1, 2), range_size(Ã)) ∘ Ã == ZeroTensor((1, 2), domain_size(Ã))
        @test Ã ∘ ZeroTensor(domain_size(Ã), (1, 2)) == ZeroTensor(range_size(Ã), (1, 2))
        @test ZeroTensor((1, 2), range_size(Ã)) ∘ ZeroTensor(range_size(Ã), (4,)) == ZeroTensor((1, 2), (4,))
        @test IdentityTensor(range_size(Ã)) ∘ ZeroTensor(range_size(Ã), (4,)) == ZeroTensor(range_size(Ã), (4,))
        @test ZeroTensor((1, 2), domain_size(Ã)) ∘ IdentityTensor(domain_size(Ã)) == ZeroTensor((1, 2), domain_size(Ã))
        @test (Ã ∘ B̃) ∘ ZeroTensor(domain_size(B̃), (1, 2)) == ZeroTensor(range_size(Ã), (1, 2))
        @test_throws DomainSizeMismatch Ã ∘ ZeroTensor(range_size(Ã), (1, 2))
    end
end

@testset "outer product operator" begin
    A = ScalingTensor(2.0, (5,))
    B = ScalingTensor(3.0, (3,))
    C = ScalingTensor(5.0, (3, 2))

    @test A ⊗ B == outer_product(A, B)
    @test A ⊗ B ⊗ C == outer_product(A, B, C)

    Ã = DenseTensor(rand(3, 2), (1,), (2,))

    @testset "IdentityTensor arguments" begin
        @test IdentityTensor(3, 2) ⊗ IdentityTensor(1, 2) == IdentityTensor(3, 2, 1, 2)
        @test IdentityTensor(3, 2) ⊗ Ã == InflatedTensor(IdentityTensor(3, 2), Ã)
        @test Ã ⊗ IdentityTensor(3, 2) == InflatedTensor(Ã, IdentityTensor(3, 2))

        I1 = IdentityTensor(3, 2)
        I2 = IdentityTensor(4)
        @test I1 ⊗ Ã ⊗ I2 == InflatedTensor(I1, Ã, I2)
    end

    @testset "ZeroTensor arguments" begin
        @test ZeroTensor(3, 2) ⊗ ZeroTensor(1, 2) == ZeroTensor(3, 2, 1, 2)
        @test ZeroTensor(3, 2) ⊗ Ã == ZeroTensor((3, 2, 3), (3, 2, 2))
        @test Ã ⊗ ZeroTensor(3, 2) == ZeroTensor((3, 3, 2), (2, 3, 2))
        @test ZeroTensor(3, 2) ⊗ IdentityTensor(1, 2) == ZeroTensor(3, 2, 1, 2)
        @test IdentityTensor(3, 2) ⊗ ZeroTensor(1, 2) == ZeroTensor(3, 2, 1, 2)
    end
end
