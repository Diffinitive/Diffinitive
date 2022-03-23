using Test
using Sbplib.LazyTensors

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
