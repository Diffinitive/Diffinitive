using Test
using Diffinitive.LazyTensors

using Tullio

struct SizeDoublingMapping{R,D} <: LazyTensor{R,D}
    domain_size::NTuple{D,Int}
end

LazyTensors.apply(m::SizeDoublingMapping{R}, v, i::Vararg{Any,R}) where {R} = (:apply, v, i)
LazyTensors.range_size(m::SizeDoublingMapping) = 2 .* m.domain_size
LazyTensors.domain_size(m::SizeDoublingMapping) = m.domain_size

@testset "TensorApplication" begin
    m = SizeDoublingMapping{1, 1}((3,))
    mm = SizeDoublingMapping{1, 1}((6,))
    v = [0, 1, 2]
    mv = TensorApplication(m, v)
    mmv = TensorApplication(mm, mv)

    @test size(mv) == 2 .* size(v)
    @test mv[1] == (:apply, v, (1,))
    @test mmv[1] == (:apply, mv, (1,))
    @test mmv[3] == (:apply, mv, (3,))
    @test mmv[6] == (:apply, mv, (6,))

    @test mv[CartesianIndex(2)] == (:apply, v, (2,))
    @test mmv[CartesianIndex(2)] == (:apply, mv, (2,))

    m = SizeDoublingMapping{2, 2}((3, 3))
    mm = SizeDoublingMapping{2, 2}((6, 6))
    v = ones(3, 3)
    mv = TensorApplication(m, v)
    mmv = TensorApplication(mm, mv)

    @test size(mv) == 2 .* size(v)
    @test mv[1, 2] == (:apply, v, (1, 2))
    @test mv[CartesianIndex(2, 3)] == (:apply, v, (2, 3))
    @test mmv[CartesianIndex(4, 3)] == (:apply, mv, (4, 3))

    m = ScalingTensor(2, (3,))
    v = [1, 2, 3]
    mv = TensorApplication(m, v)
    @test mv isa AbstractVector
    @test mv == [2, 4, 6]

    m = ScalingTensor(2, (2, 2))
    v = [[1 2]; [3 4]]
    mv = TensorApplication(m, v)
    @test mv == [[2 4]; [6 8]]
    @test mv[2, 1] == 6

    @testset "Error on index out of bounds" begin
        m = SizeDoublingMapping{1, 1}((3,))
        v = [0, 1, 2]
        mv = TensorApplication(m, v)

        @test_throws BoundsError mv[0]
        @test_throws BoundsError mv[7]
    end

    @testset "Error on unmatched dimensions" begin
        m = SizeDoublingMapping{2, 1}((3,))
        @test_throws MethodError TensorApplication(m, ones(Int, 2, 2))
    end

    @testset "Error on unmatched sizes" begin
        @test_throws DomainSizeMismatch TensorApplication(ScalingTensor(2, (2,)), ones(3))
        @test_throws DomainSizeMismatch TensorApplication(
            ScalingTensor(2, (2,)),
            TensorApplication(ScalingTensor(2, (3,)), ones(3)),
        )
    end

    @testset "Type calculation" begin
        m = ScalingTensor(2, (3,))
        v = [1.0, 2.0, 3.0]
        mv = TensorApplication(m, v)
        @test mv isa AbstractVector{Float64}
        @test mv == [2.0, 4.0, 6.0]
        @inferred TensorApplication(m, v)
        @inferred mv[1]

        m = ScalingTensor(2, (2, 2))
        v = [[1.0 2.0]; [3.0 4.0]]
        mv = TensorApplication(m, v)
        @test mv == [[2.0 4.0]; [6.0 8.0]]
        @test mv[2, 1] == 6.0
        @inferred TensorApplication(m, v)
        @inferred mv[1]

        m = ScalingTensor(2.0 + 2.0im, (3,))
        v = [1.0, 2.0, 3.0]
        mv = TensorApplication(m, v)
        @test mv isa AbstractVector{ComplexF64}
        @test mv == [2.0 + 2.0im, 4.0 + 4.0im, 6.0 + 6.0im]
        @inferred TensorApplication(m, v)
        @inferred mv[1]

        m = ScalingTensor(1, (3,))
        v = [2.0 + 2.0im, 4.0 + 4.0im, 6.0 + 6.0im]
        mv = TensorApplication(m, v)
        @test mv isa AbstractVector{ComplexF64}
        @test mv == [2.0 + 2.0im, 4.0 + 4.0im, 6.0 + 6.0im]
        @inferred TensorApplication(m, v)
        @inferred mv[1]

        m = ScalingTensor(2.0, (3,))
        v = [[1, 2, 3], [3, 2, 1], [1, 3, 1]]
        mv = TensorApplication(m, v)
        @test mv isa AbstractVector{Vector{Float64}}
        @test mv == [[2.0, 4.0, 6.0], [6.0, 4.0, 2.0], [2.0, 6.0, 2.0]]
        @inferred TensorApplication(m, v)
        @inferred mv[1]
    end

    @testset "Base.:(==)" begin
        s = [1.0, 2.0, 3.0, 4.0, 5.0]

        D = DiagonalTensor(s)
        v = rand(5)

        @test TensorApplication(D, v) == TensorApplication(D, v)
        @test TensorApplication(D, copy(v)) == TensorApplication(D, v)
        @test TensorApplication(D, 2v) != TensorApplication(D, v)
    end
end

@testset "TensorNegation" begin
    A = rand(2, 3)
    B = rand(3, 4)

    Ã = DenseTensor(A, (1,), (2,))
    B̃ = DenseTensor(B, (1,), (2,))

    negÃ = TensorNegation(Ã)
    negB̃ = TensorNegation(B̃)

    @test negÃ isa TensorNegation

    v = rand(3)
    @test TensorApplication(negÃ, v) == -TensorApplication(Ã, v)

    v = rand(4)
    @test TensorApplication(negB̃, v) == -TensorApplication(B̃, v)

    v = rand(2)
    @test TensorApplication(negÃ', v) == -TensorApplication(Ã', v)

    v = rand(3)
    @test TensorApplication(negB̃', v) == -TensorApplication(B̃', v)

    @test domain_size(negÃ) == (3,)
    @test domain_size(negB̃) == (4,)

    @test range_size(negÃ) == (2,)
    @test range_size(negB̃) == (3,)

    @testset "Base.:(==)" begin
        s = [4.0, 6.0, 5.0, 6.0, 7.0]

        @test TensorNegation(DiagonalTensor(s)) == TensorNegation(DiagonalTensor(s))
        @test TensorNegation(DiagonalTensor(copy(s))) == TensorNegation(DiagonalTensor(s))
        @test TensorNegation(DiagonalTensor(2s)) != TensorNegation(DiagonalTensor(s))
    end
end

@testset "TensorSum" begin
    A = ScalingTensor(2.0, (3,))
    B = ScalingTensor(3.0, (3,))
    AsumB = TensorSum(A, B)
    AsubB = TensorSum(A, TensorNegation(B))
    singleA = TensorSum(A)

    v = [1.1, 1.2, 1.3]
    for i in eachindex(v)
        @test TensorApplication(AsumB, v)[i] == 2v[i] + 3v[i]
        @test TensorApplication(AsubB, v)[i] == 2v[i] - 3v[i]
        @test TensorApplication(AsumB', v)[i] == 2v[i] + 3v[i]
        @test TensorApplication(singleA, v)[i] == 2v[i]
    end

    @test range_size(AsumB) == range_size(A) == range_size(B)
    @test domain_size(AsumB) == domain_size(A) == domain_size(B)
    @test TensorApplication(AsumB, ComplexF64[1.1, 1.2, 1.3])[3] isa ComplexF64

    @testset "Error on unmatched sizes" begin
        @test_throws Union{DomainSizeMismatch, RangeSizeMismatch} TensorSum(
            ScalingTensor(2.0, (3,)),
            ScalingTensor(2.0, (4,)),
        )

        @test_throws DomainSizeMismatch TensorSum(ScalingTensor(2.0, (4,)), SizeDoublingMapping{1, 1}((2,)))
        @test_throws DomainSizeMismatch TensorSum(SizeDoublingMapping{1, 1}((2,)), ScalingTensor(2.0, (4,)))
        @test_throws RangeSizeMismatch TensorSum(ScalingTensor(2.0, (2,)), SizeDoublingMapping{1, 1}((2,)))
        @test_throws RangeSizeMismatch TensorSum(SizeDoublingMapping{1, 1}((2,)), ScalingTensor(2.0, (2,)))
    end

    @testset "Constructor nesting" begin
        A = ScalingTensor(1.0, (3,))
        B = ScalingTensor(2.0, (3,))
        C = ScalingTensor(3.0, (3,))
        D = ScalingTensor(4.0, (3,))

        @test TensorSum(A, B, C, D) != TensorSum(TensorSum(A, B), TensorSum(C, D))
        @test length(TensorSum(A, B, C, D).ts) == 4

        v = rand(3)
        @test TensorApplication(TensorSum(A, B, TensorNegation(C), D), v) == 1v + 2v - 3v + 4v
        @test TensorApplication(TensorSum(TensorNegation(A), TensorNegation(B), TensorNegation(C), TensorNegation(D)), v) ==
            -1v - 2v - 3v - 4v
    end

    @testset "Base.:(==)" begin
        s = [4.0, 6.0, 5.0, 6.0, 7.0]

        @test TensorSum(DiagonalTensor(s), ScalingTensor(2.0, (5,))) == TensorSum(DiagonalTensor(s), ScalingTensor(2.0, (5,)))
        @test TensorSum(ScalingTensor(3.0, (5,)), DiagonalTensor(2s)) == TensorSum(ScalingTensor(3.0, (5,)), DiagonalTensor(2s))

        @test TensorSum(DiagonalTensor(2s), ScalingTensor(2.0, (5,))) != TensorSum(DiagonalTensor(s), ScalingTensor(2.0, (5,)))
        @test TensorSum(ScalingTensor(2.0, (5,)), DiagonalTensor(2s)) != TensorSum(ScalingTensor(3.0, (5,)), DiagonalTensor(2s))
    end
end

@testset "TensorComposition" begin
    A = rand(2, 3)
    B = rand(3, 4)

    Ã = DenseTensor(A, (1,), (2,))
    B̃ = DenseTensor(B, (1,), (2,))

    AB = TensorComposition(Ã, B̃)
    @test AB isa TensorComposition
    @test range_size(AB) == (2,)
    @test domain_size(AB) == (4,)
    @test_throws DomainSizeMismatch TensorComposition(B̃, Ã)

    # @test @inbounds TensorComposition(B̃, Ã) # Should not error even though dimensions don't match. (Since ]test runs with forced boundschecking this is currently not testable 2020-10-16)

    v = rand(4)
    @test TensorApplication(AB, v) ≈ A * B * v rtol=1e-14

    v = rand(2)
    @test TensorApplication(AB', v) ≈ B' * A' * v rtol=1e-14

    @test TensorApplication(AB, ComplexF64[1.0, 2.0, 3.0, 4.0])[1] isa ComplexF64
    @test TensorApplication(AB', ComplexF64[1.0, 2.0])[1] isa ComplexF64

    a = 2.0
    v = rand(3)
    scaledA = TensorComposition(ScalingTensor(a, range_size(Ã)), Ã)
    @test scaledA isa TensorComposition
    @test range_size(scaledA) == range_size(Ã)
    @test domain_size(scaledA) == domain_size(Ã)
    @test TensorApplication(scaledA, v) ≈ a .* A * v rtol=1e-14

    @testset "Base.:(==)" begin
        s = [4.0, 6.0, 5.0, 6.0, 7.0]

        @test TensorComposition(DiagonalTensor(s), ScalingTensor(2.0, (5,))) == TensorComposition(DiagonalTensor(s), ScalingTensor(2.0, (5,)))
        @test TensorComposition(ScalingTensor(3.0, (5,)), DiagonalTensor(2s)) == TensorComposition(ScalingTensor(3.0, (5,)), DiagonalTensor(2s))

        @test TensorComposition(DiagonalTensor(2s), ScalingTensor(2.0, (5,))) != TensorComposition(DiagonalTensor(s), ScalingTensor(2.0, (5,)))
        @test TensorComposition(ScalingTensor(2.0, (5,)), DiagonalTensor(2s)) != TensorComposition(ScalingTensor(3.0, (5,)), DiagonalTensor(2s))
    end
end

@testset "InflatedTensor" begin
    I(sz...) = IdentityTensor(sz...)

    Ã = rand(4, 2)
    B̃ = rand(4, 2, 3)
    C̃ = rand(4, 2, 3)

    A = DenseTensor(Ã, (1,), (2,))
    B = DenseTensor(B̃, (1, 2), (3,))
    C = DenseTensor(C̃, (1,), (2, 3))

    @testset "Constructors" begin
        @test InflatedTensor(I(3, 2), A, I(4)) isa LazyTensor{4, 4}
        @test InflatedTensor(I(3, 2), B, I(4)) isa LazyTensor{5, 4}
        @test InflatedTensor(I(3), C, I(2, 3)) isa LazyTensor{4, 5}
        @test InflatedTensor(C, I(2, 3)) isa LazyTensor{3, 4}
        @test InflatedTensor(I(3), C) isa LazyTensor{2, 3}
        @test InflatedTensor(I(3), I(2, 3)) isa LazyTensor{3, 3}
    end

    @testset "Range and domain size" begin
        @test range_size(InflatedTensor(I(3, 2), A, I(4))) == (3, 2, 4, 4)
        @test domain_size(InflatedTensor(I(3, 2), A, I(4))) == (3, 2, 2, 4)

        @test range_size(InflatedTensor(I(3, 2), B, I(4))) == (3, 2, 4, 2, 4)
        @test domain_size(InflatedTensor(I(3, 2), B, I(4))) == (3, 2, 3, 4)

        @test range_size(InflatedTensor(I(3), C, I(2, 3))) == (3, 4, 2, 3)
        @test domain_size(InflatedTensor(I(3), C, I(2, 3))) == (3, 2, 3, 2, 3)

        @inferred range_size(InflatedTensor(I(3, 2), A, I(4))) == (3, 2, 4, 4)
        @inferred domain_size(InflatedTensor(I(3, 2), A, I(4))) == (3, 2, 2, 4)
    end

    @testset "Application" begin
        cases = [
            (
                InflatedTensor(I(3, 2), A, I(4)),
                (v -> @tullio res[a, b, c, d] := Ã[c, i] * v[a, b, i, d]),
                (v -> @tullio res[a, b, c, d] := Ã[i, c] * v[a, b, i, d]),
            ),
            (
                InflatedTensor(I(3, 2), B, I(4)),
                (v -> @tullio res[a, b, c, d, e] := B̃[c, d, i] * v[a, b, i, e]),
                (v -> @tullio res[a, b, c, d] := B̃[i, j, c] * v[a, b, i, j, d]),
            ),
            (
                InflatedTensor(I(3, 2), C, I(4)),
                (v -> @tullio res[a, b, c, d] := C̃[c, i, j] * v[a, b, i, j, d]),
                (v -> @tullio res[a, b, c, d, e] := C̃[i, c, d] * v[a, b, i, e]),
            ),
            (
                InflatedTensor(I(3, 2), A),
                (v -> @tullio res[a, b, c] := Ã[c, i] * v[a, b, i]),
                (v -> @tullio res[a, b, c] := Ã[i, c] * v[a, b, i]),
            ),
            (
                InflatedTensor(I(3, 2), B),
                (v -> @tullio res[a, b, c, d] := B̃[c, d, i] * v[a, b, i]),
                (v -> @tullio res[a, b, c] := B̃[i, j, c] * v[a, b, i, j]),
            ),
            (
                InflatedTensor(I(3, 2), C),
                (v -> @tullio res[a, b, c] := C̃[c, i, j] * v[a, b, i, j]),
                (v -> @tullio res[a, b, c, d] := C̃[i, c, d] * v[a, b, i]),
            ),
            (
                InflatedTensor(A, I(4)),
                (v -> @tullio res[a, b] := Ã[a, i] * v[i, b]),
                (v -> @tullio res[a, b] := Ã[i, a] * v[i, b]),
            ),
            (
                InflatedTensor(B, I(4)),
                (v -> @tullio res[a, b, c] := B̃[a, b, i] * v[i, c]),
                (v -> @tullio res[a, b] := B̃[i, j, a] * v[i, j, b]),
            ),
            (
                InflatedTensor(C, I(4)),
                (v -> @tullio res[a, b] := C̃[a, i, j] * v[i, j, b]),
                (v -> @tullio res[a, b, c] := C̃[i, a, b] * v[i, c]),
            ),
        ]

        @testset "$tm" for (tm, true_apply, true_transposed_apply) in cases
            v = rand(domain_size(tm)...)
            @test TensorApplication(tm, v) ≈ true_apply(v) rtol=1e-14

            v = rand(range_size(tm)...)
            @test TensorApplication(tm', v) ≈ true_transposed_apply(v) rtol=1e-14
        end

        @testset "application to other type" begin
            tm = InflatedTensor(I(3, 2), A, I(4))

            v = rand(ComplexF64, domain_size(tm)...)
            @test TensorApplication(tm, v)[1, 2, 3, 1] isa ComplexF64

            v = rand(ComplexF64, domain_size(tm')...)
            @test TensorApplication(tm', v)[1, 2, 2, 1] isa ComplexF64
        end

        @testset "Inference of application" begin
            tm = InflatedTensor(I(2, 3), ScalingTensor(2.0, (3, 2)), I(3, 4))
            v = rand(domain_size(tm)...)
            tmv = TensorApplication(tm, v)

            @inferred apply(tm, v, 1, 2, 3, 2, 2, 4)
            @inferred tmv[1, 2, 3, 2, 2, 4]
        end
    end

    @testset "InflatedTensor of InflatedTensor" begin
        A = ScalingTensor(2.0, (2, 3))
        itm = InflatedTensor(I(3, 2), A, I(4))
        @test InflatedTensor(I(4), itm, I(2)) == InflatedTensor(I(4, 3, 2), A, I(4, 2))
        @test InflatedTensor(itm, I(2)) == InflatedTensor(I(3, 2), A, I(4, 2))
        @test InflatedTensor(I(4), itm) == InflatedTensor(I(4, 3, 2), A, I(4))

        @test InflatedTensor(I(2), I(2), I(2)) isa InflatedTensor
    end

    @testset "Base.:(==)" begin
        s = [4.0, 6.0, 5.0, 6.0, 7.0]

        @test InflatedTensor(I(1, 1), DiagonalTensor(s), I(2, 2)) == InflatedTensor(I(1, 1), DiagonalTensor(s), I(2, 2))
        @test InflatedTensor(I(1, 1), DiagonalTensor(copy(s)), I(2, 2)) == InflatedTensor(I(1, 1), DiagonalTensor(s), I(2, 2))

        @test InflatedTensor(I(1, 1), DiagonalTensor(2s), I(2, 2)) != InflatedTensor(I(1, 1), DiagonalTensor(s), I(2, 2))
        @test InflatedTensor(I(1, 2), DiagonalTensor(s), I(2, 2)) != InflatedTensor(I(1, 1), DiagonalTensor(s), I(2, 2))
        @test InflatedTensor(I(1, 1), DiagonalTensor(s), I(2, 3)) != InflatedTensor(I(1, 1), DiagonalTensor(s), I(2, 2))
    end
end

@testset "outer_product" begin
    A = ScalingTensor(2.0, (5,))
    B = ScalingTensor(3.0, (3,))
    C = ScalingTensor(5.0, (3, 2))

    AB = outer_product(A, B)
    @test AB isa LazyTensor{2, 2}
    @test range_size(AB) == (5, 3)
    @test domain_size(AB) == (5, 3)

    v = rand(range_size(AB)...)
    @test TensorApplication(AB, v) == 6v

    ABC = outer_product(A, B, C)
    @test ABC isa LazyTensor{4, 4}
    @test range_size(ABC) == (5, 3, 3, 2)
    @test domain_size(ABC) == (5, 3, 3, 2)

    A = rand(3, 2)
    B = rand(2, 4, 3)

    v₁ = rand(2, 4, 3)
    v₂ = rand(4, 3, 2)

    Ã = DenseTensor(A, (1,), (2,))
    B̃ = DenseTensor(B, (1,), (2, 3))

    ÃB̃ = outer_product(Ã, B̃)
    @tullio ABv[i, k] := A[i, j] * B[k, l, m] * v₁[j, l, m]
    @test TensorApplication(ÃB̃, v₁) ≈ ABv

    B̃Ã = outer_product(B̃, Ã)
    @tullio BAv[k, i] := A[i, j] * B[k, l, m] * v₂[l, m, j]
    @test TensorApplication(B̃Ã, v₂) ≈ BAv
end

@testset "inflate" begin
    I = LazyTensors.inflate(IdentityTensor(), (3, 4, 5, 6), 2)
    @test I isa LazyTensor{3, 3}
    @test range_size(I) == (3, 5, 6)
    @test domain_size(I) == (3, 5, 6)

    @test LazyTensors.inflate(ScalingTensor(1.0, (4,)), (3, 4, 5, 6), 1) ==
        InflatedTensor(IdentityTensor(), ScalingTensor(1.0, (4,)), IdentityTensor(4, 5, 6))
    @test LazyTensors.inflate(ScalingTensor(2.0, (1,)), (3, 4, 5, 6), 2) ==
        InflatedTensor(IdentityTensor(3), ScalingTensor(2.0, (1,)), IdentityTensor(5, 6))
    @test LazyTensors.inflate(ScalingTensor(3.0, (6,)), (3, 4, 5, 6), 4) ==
        InflatedTensor(IdentityTensor(3, 4, 5), ScalingTensor(3.0, (6,)), IdentityTensor())

    @test_throws BoundsError LazyTensors.inflate(ScalingTensor(1.0, (4,)), (3, 4, 5, 6), 0)
    @test_throws BoundsError LazyTensors.inflate(ScalingTensor(1.0, (4,)), (3, 4, 5, 6), 5)
end
