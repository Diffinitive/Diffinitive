using Test
using Diffinitive.LazyTensors
using Diffinitive.LazyTensors: TupleTable, tuple_range
using BenchmarkTools

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
    A = DenseTensor(Ã,(1,),(2,))
    I1 = IdentityTensor{Float64}(2)
    I2 = IdentityTensor{Float64}(4)
    @test A∘I1 == A
    @test I2∘A == A
    @test I1∘I1 == I1
    @test_throws DomainSizeMismatch I1∘A
    @test_throws DomainSizeMismatch A∘I2
    @test_throws DomainSizeMismatch I1∘I2
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

@testset "DiagonalTensor" begin
    @test DiagonalTensor([1,2,3,4]) isa LazyTensor{Int,1,1}
    @test DiagonalTensor([1 2 3; 4 5 6]) isa LazyTensor{Int,2,2}
    @test DiagonalTensor([1. 2. 3.; 4. 5. 6.]) isa LazyTensor{Float64,2,2}

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
end


@testset "DenseTensor" begin
    # Test a standard matrix-vector product
    # mapping vectors of size 4 to vectors of size 3.
    A = rand(3,4)
    Ã = DenseTensor(A, (1,), (2,))
    v = rand(4)
    w = rand(3)

    @test Ã isa DenseTensor{T,1,1} where T
    @test Ã isa LazyTensor{T,1,1} where T
    @test range_size(Ã) == (3,)
    @test domain_size(Ã) == (4,)

    @test Ã*ones(4) ≈ A*ones(4) atol=5e-13
    @test Ã*v ≈ A*v atol=5e-13
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
    @test B̃ isa LazyTensor{T,2,1} where T
    @test B̃*ones(2) ≈ B[:,:,1] + B[:,:,2] atol=5e-13
    @test B̃*v ≈ B[:,:,1]*v[1] + B[:,:,2]*v[2] atol=5e-13

    # Map matrices of size (3,2) to vectors of size 4
    B̃ = DenseTensor(B, (2,), (1,3))
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

@testset "TupleTable" begin
    @testset "Constructors" begin
        @test TupleTable((1,2,3)) isa TupleTable{1,3}
        @test TupleTable((1,2),(3,4)) isa TupleTable{2,2}
        @test TupleTable((1,2,3),(3,4,5)) isa TupleTable{2,3}
        @test TupleTable((1,2),(3,4),(5,6)) isa TupleTable{3,2}

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
end

@testset "VectorTensor" begin
    s = [4., 6, 5, 6, 7]
    t = VectorTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
    v = rand(5)
    expected = map((sᵢ, vᵢ)-> sᵢ*vᵢ+3vᵢ, s,v)
    @test t*v == expected
end

@testset "VectorDotTensor" begin
    s = [4., 6, 5, 6, 7]
    t = VectorDotTensor(DiagonalTensor(s), ScalingTensor(3., (5,)))
    v = rand(SVector{2,Float64}, 5)
    expected = map((sᵢ, vᵢ)-> sᵢ*vᵢ[1]+3vᵢ[2], s,v)
    @test t*v == expected
end

@testset "MatrixTensor" begin

end

@testset "tuple_range()" begin
    @test tuple_range(1) == (1,)
    @test tuple_range(2) == (1,2)
    @test tuple_range(5) == (1,2,3,4,5)

    @test tuple_range(Val(1)) == (1,)
    @test tuple_range(Val(2)) == (1,2)
    @test tuple_range(Val(5)) == (1,2,3,4,5)
end
