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
    @test_throws DomainSizeMismatch IdentityTensor(4) * v

    @testset "Error hint" begin
        err = try
            m * m
        catch err
            err
        end
        @test occursin("Did you mean to use `∘` to compose lazy tensors?", sprint(showerror, err))
    end
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
    @test -(-A) == A

    @testset "ZeroTensor arguments" begin
        A = ScalingTensor(1.0, (3, 3))
        B = ScalingTensor(2.0, (3, 3))
        Z =  ZeroTensor(3, 3)

        @test Z + A == A + Z == A
        @test Z - A == -A
        @test A - Z == A
        @test A + Z + B == A + B
        @test Z + A + Z == A
        @test Z + Z == Z
        @test (A+B) + Z == Z + (A+B) == A+B
        @test -Z == Z
        @test_throws DomainSizeMismatch ZeroTensor((3, 3), (2, 3)) + A
        @test_throws RangeSizeMismatch ZeroTensor((1, 3), (3, 3)) + A
    end
end

@testset "composition operator" begin
    A = rand(2, 3)
    B = rand(3, 4)
    C = rand(4, 5)
    D = rand(5, 6)

    Ã = DenseTensor(A, (1,), (2,))
    B̃ = DenseTensor(B, (1,), (2,))
    C̃ = DenseTensor(C, (1,), (2,))
    D̃ = DenseTensor(D, (1,), (2,))

    @test Ã ∘ B̃ == TensorComposition(Ã, B̃)
    @test_throws DomainSizeMismatch B̃ ∘ Ã

    @test (Ã ∘ B̃) ∘ C̃ == TensorComposition(Ã, TensorComposition(B̃, C̃))

    @test (Ã ∘ B̃) ∘ (C̃∘D̃) == TensorComposition(Ã, TensorComposition(B̃, TensorComposition(C̃,D̃)))

    @testset "Non-LazyTensor arguments" begin
        @test_throws MethodError Ã ∘ sin
        @test_throws MethodError sin ∘ Ã
        @test_throws MethodError Ã ∘ :not_a_lazy_tensor
        @test_throws MethodError :not_a_lazy_tensor ∘ Ã
    end

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

@testset "Base.:+ for VectorTensor" begin
    s = [4., 6., 5., 6., 7.]
    A  = VectorTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
    B  = VectorTensor(ScalingTensor(2., (5,)), DiagonalTensor(2s))
    C = VectorTensor(ScalingTensor(2., (5,)), ScalingTensor(3., (5,)), ScalingTensor(2., (5,)))
    D = VectorTensor(ZeroTensor((5,), (1,)), ZeroTensor((5,), (1,)))
    E = VectorTensor(ZeroTensor((1,), (5,)), ZeroTensor((1,), (5,)))

    ApB = VectorTensor(
        DiagonalTensor(s) + ScalingTensor(2., (5,)),
        ScalingTensor(3., (5,)) + DiagonalTensor(2s),
    )

    @test A+B == ApB
    @test_throws DimensionMismatch A + C
    @test_throws DomainSizeMismatch A + D
    @test_throws RangeSizeMismatch A + E
end

@testset "Base.:+ for VectorDotTensor" begin
    s = [4., 6., 5., 6., 7.]
    A  = VectorDotTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
    B  = VectorDotTensor(ScalingTensor(2., (5,)), DiagonalTensor(2s))
    C = VectorDotTensor(ScalingTensor(2., (5,)), ScalingTensor(3., (5,)), ScalingTensor(2., (5,)))
    D = VectorDotTensor(ZeroTensor((5,), (1,)), ZeroTensor((5,), (1,)))
    E = VectorDotTensor(ZeroTensor((1,), (5,)), ZeroTensor((1,), (5,)))

    ApB = VectorDotTensor(
        DiagonalTensor(s) + ScalingTensor(2., (5,)),
        ScalingTensor(3., (5,)) + DiagonalTensor(2s),
    )

    @test A+B == ApB
    @test_throws DimensionMismatch A + C
    @test_throws DomainSizeMismatch A + D
    @test_throws RangeSizeMismatch A + E
end

@testset "Base.:+ for MatrixTensor" begin
    s1 = [4., 6., 5., 6., 7.]
    s2 = [6., 9., 3., 5., 7.]
    A  = MatrixTensor(
        (DiagonalTensor(s1), ScalingTensor(3.,(5,))),
        (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
    )

    B = MatrixTensor(
        (ScalingTensor(5.,(5,)), DiagonalTensor(2s1)),
        (DiagonalTensor(2s2), ScalingTensor(7., (5,))),
    )
    C = MatrixTensor(
        (ScalingTensor(5.,(5,)), DiagonalTensor(2s1)),
        (DiagonalTensor(2s2), ScalingTensor(7., (5,))),
        (DiagonalTensor(2s2), ScalingTensor(7., (5,))))
    D = MatrixTensor(
        (ZeroTensor((5,), (1,)), ZeroTensor((5,), (1,))),
        (ZeroTensor((5,), (1,)), ZeroTensor((5,), (1,)))
    )
    E = MatrixTensor(
        (ZeroTensor((1,), (5,)), ZeroTensor((1,), (5,))),
        (ZeroTensor((1,), (5,)), ZeroTensor((1,), (5,)))
    )

    ApB = MatrixTensor(
        (DiagonalTensor(s1)+ScalingTensor(5.,(5,)), ScalingTensor(3.,(5,))+ DiagonalTensor(2s1)),
        (ScalingTensor(6., (5,))+DiagonalTensor(2s2), DiagonalTensor(s2) + ScalingTensor(7., (5,))),
    )

    @test A+B == ApB
    @test_throws DimensionMismatch A + C
    @test_throws DomainSizeMismatch A + D
    @test_throws RangeSizeMismatch A + E
end
