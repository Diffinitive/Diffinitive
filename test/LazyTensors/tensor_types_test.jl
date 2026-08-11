using Test
using Diffinitive.LazyTensors
using Diffinitive.LazyTensors: TupleTable, tuple_range
using BenchmarkTools
using StaticArrays

@testset "IdentityTensor" begin
    @test IdentityTensor((4,5)) isa IdentityTensor{2}
    @test IdentityTensor((4,5)) isa LazyTensor{2,2}
    @test IdentityTensor((4,5)) == IdentityTensor(4,5)

    @test IdentityTensor(3,2) isa IdentityTensor{2}

    for sz ∈ [(4,5),(3,),(5,6,4)]
        I = IdentityTensor(sz)
        v = rand(sz...)
        @test I*v == v
        @test I'*v == v

        v = rand(ComplexF64,sz...)
        @test I*v == v
        @test I'*v == v

        @test range_size(I) == sz
        @test domain_size(I) == sz
    end

    I = IdentityTensor((4,5))
    v = rand(4,5)
    @inferred (I*v)[3,2]
    @inferred (I'*v)[3,2]
    @inferred range_size(I)

    @inferred range_dim(I)
    @inferred domain_dim(I)

    Ã = rand(4,2)
    A = DenseTensor(Ã,(1,),(2,))
    I1 = IdentityTensor(2)
    I2 = IdentityTensor(4)
    @test A∘I1 == A
    @test I2∘A == A
    @test I1∘I1 == I1
    @test_throws DomainSizeMismatch I1∘A
    @test_throws DomainSizeMismatch A∘I2
    @test_throws DomainSizeMismatch I1∘I2


     @testset "Base.:(==)" begin
        @test IdentityTensor(3,2) == IdentityTensor(3,2)

        @test IdentityTensor(3,2,4) != IdentityTensor(3,2)
        @test IdentityTensor(3) != IdentityTensor(3,2)
        @test IdentityTensor(2,2) != IdentityTensor(3,2)
    end
end

@testset "ZeroTensor" begin
    @testset "Constructors" begin
        @test ZeroTensor((2,3),(3,4)) isa LazyTensor{2,2}
        @test zero(ScalingTensor(1, (10,4))) == ZeroTensor((10,4),(10,4))

        @test ZeroTensor(3,5,6) == ZeroTensor((3,5,6),(3,5,6))
        @test ZeroTensor((3,5,6)) == ZeroTensor((3,5,6),(3,5,6))

        B = rand(3,4,2)
        B̃ = DenseTensor(B, (1,2), (3,))
        @test zero(B̃) == ZeroTensor((3,4),(2,))
    end

    @testset "range_size" begin
        @test range_size(ZeroTensor((1,2),(3,4))) == (1,2)
        @test range_size(ZeroTensor((11,),(3,4))) == (11,)
    end

    @testset "domain_size" begin
        @test domain_size(ZeroTensor((1,2),(3,4))) == (3,4)
        @test domain_size(ZeroTensor((11,),(3,4,5))) == (3,4,5)
    end

    @testset "Application" begin
        v = rand(4,5)
        @test ZeroTensor(4,5)*v isa AbstractArray{Float64,2}
        @test ZeroTensor(4,5)*v == zeros(4,5)

        @test ZeroTensor((3,2,3),(4,5))*v isa AbstractArray{Float64,3}
        @test ZeroTensor((3,2,3),(4,5))*v == zeros(3,2,3)

        v = rand(1:100, 4,5)
        @test ZeroTensor(4,5)*v isa AbstractArray{Int,2}
        @test ZeroTensor(4,5)*v == zeros(4,5)

        @test ZeroTensor((3,2,3),(4,5))*v isa AbstractArray{Int,3}
        @test ZeroTensor((3,2,3),(4,5))*v == zeros(3,2,3)
    end

    @testset "Composition" begin
        @test ZeroTensor((3,4),(5,6))∘ZeroTensor((5,6),(5,4)) == ZeroTensor((3,4),(5,4))
        @test ZeroTensor((4,),(3,2))∘ZeroTensor((3,2),(5,4,2)) == ZeroTensor((4,), (5,4,2))
        @test ZeroTensor((2,3),(10,9))∘ScalingTensor(1., (10,9)) == ZeroTensor((2,3),(10,9))
        @test ZeroTensor((2,1,4),(10,9))∘ScalingTensor(1., (10,9)) == ZeroTensor((2,1,4),(10,9))
        @test ScalingTensor(1., (10,9))∘ZeroTensor((10,9),(7,8))== ZeroTensor((10,9),(7,8))
        @test ScalingTensor(1., (10,9))∘ZeroTensor((10,9),(4,2,7))== ZeroTensor((10,9),(4,2,7))
    end

    @testset "Addition" begin
        @test ZeroTensor((1,2),(3,4)) + ZeroTensor((1,2),(3,4)) == ZeroTensor((1,2),(3,4))
        @test ZeroTensor((4,3),(5,4,3)) + ZeroTensor((4,3),(5,4,3)) == ZeroTensor((4,3),(5,4,3))
        @test ZeroTensor(10,9) + ScalingTensor(1., (10,9)) == ScalingTensor(1., (10,9))
        @test ScalingTensor(1., (10,9)) + ZeroTensor(10,9) == ScalingTensor(1., (10,9))
    end
end

@testset "ScalingTensor" begin
    st = ScalingTensor(2.,(3,4))
    @test st isa LazyTensor{2, 2}
    @test range_size(st) == (3,4)
    @test domain_size(st) == (3,4)

    v = rand(3,4)
    @test st*v == 2.0 .* v
    @test st'*v == 2.0 .* v

    complex_st = ScalingTensor(2 + 3im, (2,))
    @test complex_st'*[1 - im, 4 + 2im] == [-1 - 5im, 14 - 8im]

    @inferred (st*v)[2,2]
    @inferred (st'*v)[2,2]

    @testset "Base.:(==)" begin
        @test ScalingTensor(2.,(3,4)) == ScalingTensor(2.,(3,4))
        @test ScalingTensor(2.,(3,4)) == ScalingTensor(2,(3,4))

        @test ScalingTensor(3.,(3,4)) != ScalingTensor(2,(3,4))
        @test ScalingTensor(2.,(2,4)) != ScalingTensor(2,(3,4))
        @test ScalingTensor(2.,(3,)) != ScalingTensor(2,(3,4))
        @test ScalingTensor(2.,(3,4,2)) != ScalingTensor(2,(3,4))
    end
end

@testset "DiagonalTensor" begin
    @test DiagonalTensor([1,2,3,4]) isa LazyTensor{1,1}
    @test DiagonalTensor([1 2 3; 4 5 6]) isa LazyTensor{2,2}
    @test DiagonalTensor([1. 2. 3.; 4. 5. 6.]) isa LazyTensor{2,2}

    @test range_size(DiagonalTensor([1,2,3,4])) == (4,)
    @test domain_size(DiagonalTensor([1,2,3,4])) == (4,)

    @test range_size(DiagonalTensor([1 2 3; 4 5 6])) == (2,3)
    @test domain_size(DiagonalTensor([1 2 3; 4 5 6])) == (2,3)

    @testset "apply size=$sz" for sz ∈ [(4,),(3,2),(3,4,2)]
        diag = rand(sz...)
        tm = DiagonalTensor(diag)

        v = rand(sz...)

        @test tm*v == diag.*v
        @test tm'*v == diag.*v
    end

    complex_tm = DiagonalTensor([1 + 2im, 3 - 4im])
    @test complex_tm'*[5 - 6im, 7 + 8im] == [-7 - 16im, -11 + 52im]

    @testset "allocations size=$sz" for sz ∈ [(4,),(3,2),(3,4,2)]
        diag = rand(sz...)
        tm = DiagonalTensor(diag)

        v = rand(sz...)

        @test tm*v == diag.*v
        @test tm'*v == diag.*v
    end

    sz = (3,2)
    diag = rand(sz...)
    tm = DiagonalTensor(diag)

    v = rand(sz...)
    LazyTensors.apply(tm,v, 2,1)
    @test (@ballocated LazyTensors.apply($tm,$v, 2,1)) == 0


    @testset "Base.:(==)" begin
        @test DiagonalTensor([1,2,3,4]) == DiagonalTensor([1,2,3,4])
        @test DiagonalTensor([2,2,3,4]) != DiagonalTensor([1,2,3,4])
    end
end


@testset "DenseTensor" begin
    # Test a standard matrix-vector product
    # mapping vectors of size 4 to vectors of size 3.
    A = rand(3,4)
    Ã = DenseTensor(A, (1,), (2,))
    v = rand(4)
    w = rand(3)

    @test Ã isa DenseTensor{1,1}
    @test Ã isa LazyTensor{1,1}
    @test range_size(Ã) == (3,)
    @test domain_size(Ã) == (4,)

    @test Ã*ones(4) ≈ A*ones(4) atol=5e-13
    @test Ã*v ≈ A*v atol=5e-13
    @test Ã'*w ≈ A'*w

    A = [
        1 + 2im 3 - 4im;
        5 + 6im 7 + 8im;
    ]
    Ã = DenseTensor(A, (1,), (2,))
    w = [2 - im, 3 + 2im]
    @test Ã'*w ≈ A'*w

    A = rand(2,3,4)
    @test_throws DomainError DenseTensor(A, (3,1), (2,))

    # Test more exotic mappings
    B = rand(3,4,2)
    # Map vectors of size 2 to matrices of size (3,4)
    B̃ = DenseTensor(B, (1,2), (3,))
    v = rand(2)

    @test range_size(B̃) == (3,4)
    @test domain_size(B̃) == (2,)
    @test B̃ isa LazyTensor{2,1}
    @test B̃*ones(2) ≈ B[:,:,1] + B[:,:,2] atol=5e-13
    @test B̃*v ≈ B[:,:,1]*v[1] + B[:,:,2]*v[2] atol=5e-13

    # Map matrices of size (3,2) to vectors of size 4
    B̃ = DenseTensor(B, (2,), (1,3))
    v = rand(3,2)

    @test range_size(B̃) == (4,)
    @test domain_size(B̃) == (3,2)
    @test B̃ isa LazyTensor{1,2}
    @test B̃*ones(3,2) ≈ B[1,:,1] + B[2,:,1] + B[3,:,1] +
                        B[1,:,2] + B[2,:,2] + B[3,:,2] atol=5e-13
    @test B̃*v ≈ B[1,:,1]*v[1,1] + B[2,:,1]*v[2,1] + B[3,:,1]*v[3,1] +
                B[1,:,2]v[1,2] + B[2,:,2]*v[2,2] + B[3,:,2]*v[3,2] atol=5e-13


    @testset "Base.:(==)" begin
        A = rand(2,3,4)

        @test DenseTensor(A, (1,), (2,3)) == DenseTensor(A, (1,), (2,3))
        @test DenseTensor(copy(A), (1,), (2,3)) == DenseTensor(A, (1,), (2,3))

        @test DenseTensor(2A, (1,), (2,3)) != DenseTensor(A, (1,), (2,3))
        @test DenseTensor(A, (1,2), (3,)) != DenseTensor(A, (1,), (2,3))
    end

    # TODO:
    # @inferred (B̃*v)[2]
end

@testset "TupleTable" begin
    @testset "Constructors" begin
        @test TupleTable((1,2,3)) isa TupleTable{1,3}
        @test TupleTable((1,2),(3,4)) isa TupleTable{2,2}
        @test TupleTable((1,2,3),(3,4,5)) isa TupleTable{2,3}
        @test TupleTable((1,2),(3,4),(5,6)) isa TupleTable{3,2}

        @test TupleTable([1 2; 3 4]) isa TupleTable{2,2}
        @test TupleTable([1 2 3; 3 4 5]) isa TupleTable{2,3}

        @test_throws DimensionMismatch("All rows must have the same length") TupleTable((1,2),(1,2,3))
    end

    @testset "size" begin
        @test size(TupleTable((1,2,3))) == (1,3)
        @test size(TupleTable((1,2),(3,4))) == (2,2)
        @test size(TupleTable((1,2,3),(3,4,5))) == (2,3)
        @test size(TupleTable((1,2),(3,4),(5,6))) == (3,2)

        @test size(TupleTable{1,3}) == (1,3)
        @test size(TupleTable{2,2}) == (2,2)
        @test size(TupleTable{2,3}) == (2,3)
        @test size(TupleTable{3,2}) == (3,2)
    end

    @testset "getindex" begin
        tt = TupleTable((1,2,3))
        @test tt[1,1] == 1
        @test tt[1,2] == 2
        @test tt[1,3] == 3
        @test_throws BoundsError tt[2,2]

        @test tt[1,:] == (1,2,3)
        @test_throws BoundsError tt[2,:]

        tt = TupleTable((1,2),(3,4))
        @test tt[1,1] == 1
        @test tt[1,2] == 2
        @test tt[2,1] == 3
        @test tt[2,2] == 4
        @test_throws BoundsError tt[3,2]
        @test_throws BoundsError tt[2,3]

        @test tt[1,:] == (1,2)
        @test tt[2,:] == (3,4)
        @test_throws BoundsError tt[3,:]

        tt = TupleTable((1,2,3),(3,4,5))
        @test tt[1,3] == 3
        @test tt[2,1] == 3
        @test tt[2,3] == 5

        tt = TupleTable((1,2),(3,4),(5,6))
        @test tt[1,1] == 1
        @test tt[2,2] == 4
        @test tt[3,1] == 5
    end

    @testset "Base.adjoint" begin
        tt = TupleTable((1+1im,2+2im,3+3im),(4+4im,5+5im,6+6im))
        expected = TupleTable((1-1im, 4-4im),(2-2im, 5-5im),(3-3im, 6-6im))
        @test adjoint(tt) == expected
    end

    @testset "Base.:(==)" begin
        @test TupleTable((1,2),(3,4)) == TupleTable((1,2),(3,4))
        @test TupleTable(([1,2],2),(3,4)) == TupleTable(([1,2],2),(3,4))

        @test TupleTable((2,2),(3,4)) != TupleTable((1,2),(3,4))
        @test TupleTable(([2,2],2),(3,4)) != TupleTable(([1,2],2),(3,4))
    end

    @testset "Base.:+" begin
        A = TupleTable((1,2),(3,4))
        @test A+A == TupleTable((2,4),(6,8))

        A = TupleTable((1,2),(3,4))
        B = TupleTable((3,2),(1,0))
        @test A+B == TupleTable((4,4),(4,4))
    end
end

@testset "VectorTensor" begin
    @testset "Constructors" begin
        s = [4., 6., 5., 6., 7.]
        @test VectorTensor(DiagonalTensor(s), ScalingTensor(3., (5,))) isa LazyTensor{1, 1}
        @test_throws DomainSizeMismatch VectorTensor(ZeroTensor((1,), (1,)), ZeroTensor((1,), (2,)))
        @test_throws RangeSizeMismatch VectorTensor(ZeroTensor((1,), (1,)), ZeroTensor((2,), (1,)))

        @test VectorTensor(ZeroTensor((2,), (3,4)), ZeroTensor((2,), (3,4))) isa LazyTensor{1, 2}

        A = VectorTensor(3) do i
            DiagonalTensor(i*s)
        end
        @test A == VectorTensor(DiagonalTensor(s),DiagonalTensor(2s),DiagonalTensor(3s))
    end

    @testset "apply" begin
        s = [4., 6., 5., 6., 7.]
        t = VectorTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
        v = [10., 11., 12., 14., 15.]
        expected = map((sᵢ, vᵢ)-> @SVector[sᵢ*vᵢ, 3vᵢ], s,v)
        @test t*v == expected
        @test collect(t*v) isa Vector{SVector{2,Float64}}
    end

    @testset "Base.:(==)" begin
        s = [4., 6., 5., 6., 7.]
        @test VectorTensor(DiagonalTensor(s), ScalingTensor(3., (5,))) == VectorTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
        @test VectorTensor(DiagonalTensor(2s), ScalingTensor(3., (5,))) == VectorTensor(DiagonalTensor(2s), ScalingTensor(3., (5,)))

        @test VectorTensor(DiagonalTensor(s), ScalingTensor(2., (5,))) != VectorTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
        @test VectorTensor(DiagonalTensor(2s), ScalingTensor(3., (5,))) != VectorTensor(DiagonalTensor(3s), ScalingTensor(3., (5,)))
    end

end

@testset "VectorDotTensor" begin
    @testset "Constructors" begin
        s = [4., 6., 5., 6., 7.]
        @test VectorDotTensor(DiagonalTensor(s), ScalingTensor(3., (5,))) isa LazyTensor{1, 1}
        @test_throws DomainSizeMismatch VectorDotTensor(ZeroTensor((1,), (1,)), ZeroTensor((1,), (2,)))
        @test_throws RangeSizeMismatch VectorDotTensor(ZeroTensor((1,), (1,)), ZeroTensor((2,), (1,)))

        @test VectorDotTensor(ZeroTensor((2,), (3,4)), ZeroTensor((2,), (3,4))) isa LazyTensor{1, 2}

         A = VectorDotTensor(3) do i
            DiagonalTensor(i*s)
        end
        @test A == VectorDotTensor(DiagonalTensor(s),DiagonalTensor(2s),DiagonalTensor(3s))
    end

    @testset "apply" begin
        s = [4., 6., 5., 6., 7.]
        t = VectorDotTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
        v = rand(SVector{2,Float64}, 5)
        expected = map((sᵢ, vᵢ)-> sᵢ*vᵢ[1]+3vᵢ[2], s,v)
        @test t*v == expected
        @test collect(t*v) isa Vector{Float64}
    end

    @testset "Base.:(==)" begin
        s = [4., 6., 5., 6., 7.]

        @test VectorDotTensor(DiagonalTensor(s), ScalingTensor(3., (5,))) == VectorDotTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
        @test VectorDotTensor(DiagonalTensor(2s), ScalingTensor(3., (5,))) == VectorDotTensor(DiagonalTensor(2s), ScalingTensor(3., (5,)))

        @test VectorDotTensor(DiagonalTensor(s), ScalingTensor(2., (5,))) != VectorDotTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
        @test VectorDotTensor(DiagonalTensor(3s), ScalingTensor(3., (5,))) != VectorDotTensor(DiagonalTensor(2s), ScalingTensor(3., (5,)))

    end

end

@testset "MatrixTensor" begin
    @testset "Constructors" begin
        s1 = [4., 6., 5., 6., 7.]
        s2 = [6., 9., 3., 5., 7.]
        t = MatrixTensor(
            (DiagonalTensor(s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )

        @test t isa LazyTensor{1, 1}

        @test MatrixTensor(
            (ZeroTensor((2,), (3,4)), ZeroTensor((2,), (3,4))),
            (ZeroTensor((2,), (3,4)), ZeroTensor((2,), (3,4))),
        ) isa LazyTensor{1, 2}


        t2 = MatrixTensor((
            (DiagonalTensor(s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        ))

        @test t2 isa LazyTensor{1, 1}
        @test t2 == t


        A = [
            DiagonalTensor(s1) ScalingTensor(3.,(5,));
            ScalingTensor(6., (5,)) DiagonalTensor(s2);
        ]

        @test MatrixTensor(A) == t

        TT = TupleTable(A)

        @test MatrixTensor(TT) == t


        A = MatrixTensor(3,2) do i,j
            DiagonalTensor(i*s1 + j*s2)
        end

        @test A == MatrixTensor(
            (DiagonalTensor(1s1 + 1s2), DiagonalTensor(1s1 + 2s2)),
            (DiagonalTensor(2s1 + 1s2), DiagonalTensor(2s1 + 2s2)),
            (DiagonalTensor(3s1 + 1s2), DiagonalTensor(3s1 + 2s2)),
        )

        @test_throws DomainSizeMismatch MatrixTensor(
            (DiagonalTensor(1s1 + 1s2), DiagonalTensor(1s1 + 2s2)),
            (DiagonalTensor(2s1 + 1s2), ZeroTensor((5,), (1,)))
        )
        @test_throws RangeSizeMismatch MatrixTensor(
            (DiagonalTensor(1s1 + 1s2), DiagonalTensor(1s1 + 2s2)),
            (DiagonalTensor(2s1 + 1s2), ZeroTensor((1,), (5,)))
        )
    end

    @testset "apply" begin
        s1 = [4., 6., 5., 6., 7.]
        s2 = [6., 9., 3., 5., 7.]
        t = MatrixTensor(
            (DiagonalTensor(s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )
        v = reinterpret(SVector{2,Float64}, rand(1.:20., 10))

        expected = map(s1,s2,v) do s1ᵢ, s2ᵢ, vᵢ
            @SVector[
                s1ᵢ*vᵢ[1] + 3*vᵢ[2],
                6*vᵢ[1] + s2ᵢ*vᵢ[2],
            ]
        end

        @test t*v == expected
    end

    @testset "Base.:(==)" begin
        s1 = [4., 6., 5., 6., 7.]
        s2 = [6., 9., 3., 5., 7.]
        A  = MatrixTensor(
            (DiagonalTensor(s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )
        B  = MatrixTensor(
            (DiagonalTensor(s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )
        @test A == B

        A  = MatrixTensor(
            (DiagonalTensor(2s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )
        B  = MatrixTensor(
            (DiagonalTensor(2s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )
        @test A == B


        A  = MatrixTensor(
            (DiagonalTensor(s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(5., (5,)), DiagonalTensor(s2)),
        )
        B  = MatrixTensor(
            (DiagonalTensor(s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )
        @test A != B

        A  = MatrixTensor(
            (DiagonalTensor(2s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )
        B  = MatrixTensor(
            (DiagonalTensor(3s1), ScalingTensor(3.,(5,))),
            (ScalingTensor(6., (5,)), DiagonalTensor(s2)),
        )
        @test A != B
    end

end

@testset "tuple_range()" begin
    @test tuple_range(1) == (1,)
    @test tuple_range(2) == (1,2)
    @test tuple_range(5) == (1,2,3,4,5)

    @test tuple_range(Val(1)) == (1,)
    @test tuple_range(Val(2)) == (1,2)
    @test tuple_range(Val(5)) == (1,2,3,4,5)
end
