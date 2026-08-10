using Test
using Diffinitive.LazyTensors

@testset "Generic Mapping methods" begin
    struct DummyMapping{R,D} <: LazyTensor{R,D} end
    LazyTensors.apply(m::DummyMapping{R,D}, v, I::Vararg{Any,R}) where {R,D} = :apply
    @test range_dim(DummyMapping{2,3}()) == 2
    @test domain_dim(DummyMapping{2,3}()) == 3
    @test apply(DummyMapping{2,3}(), zeros(Int, (0,0,0)),0,0) == :apply
end

@testset "Size check interface" begin
    t = ZeroTensor((2, 3), (4, 5))
    matching = ZeroTensor((2, 3), (4, 5))
    different_domain = ZeroTensor((2, 3), (4, 6))
    different_range = ZeroTensor((2, 4), (4, 5))

    @testset "check_domain_size" begin
        @test check_domain_size(Bool, t, (4, 5))
        @test !check_domain_size(Bool, t, (4, 6))
        @test isnothing(check_domain_size(t, (4, 5)))
        @test_throws DomainSizeMismatch check_domain_size(t, (4, 6))

        domain_error = DomainSizeMismatch(t, (4, 6))
        @test sprint(showerror, domain_error) ==
            "DomainSizeMismatch: domain size (4, 5) of LazyTensor not matching size (4, 6)"
    end

    @testset "check_range_size" begin
        @test check_range_size(Bool, t, (2, 3))
        @test !check_range_size(Bool, t, (2, 4))
        @test isnothing(check_range_size(t, (2, 3)))
        @test_throws RangeSizeMismatch check_range_size(t, (2, 4))

        range_error = RangeSizeMismatch(t, (2, 4))
        @test sprint(showerror, range_error) ==
            "RangeSizeMismatch: range size (2, 3) of LazyTensor not matching size (2, 4)"
    end

    @testset "check_equal_size" begin
        another_matching = ZeroTensor((2, 3), (4, 5))

        @test check_equal_size(Bool, t, matching)
        @test check_equal_size(Bool, t, matching, another_matching)
        @test !check_equal_size(Bool, t, different_domain)
        @test !check_equal_size(Bool, t, different_range)
        @test !check_equal_size(Bool, t, matching, different_domain)
        @test !check_equal_size(Bool, t, matching, different_range)
        @test isnothing(check_equal_size(t, matching))
        @test isnothing(check_equal_size(t, matching, another_matching))
        @test_throws DomainSizeMismatch check_equal_size(t, different_domain)
        @test_throws RangeSizeMismatch check_equal_size(t, different_range)
        @test_throws DomainSizeMismatch check_equal_size(t, matching, different_domain)
        @test_throws RangeSizeMismatch check_equal_size(t, matching, different_range)
    end

    @testset "check_composable" begin
        composable = ZeroTensor((4, 5), (7,))
        noncomposable = ZeroTensor((4, 6), (7,))

        @test check_composable(Bool, t, composable)
        @test !check_composable(Bool, t, noncomposable)
        @test isnothing(check_composable(t, composable))
        @test_throws DomainSizeMismatch check_composable(t, noncomposable)
    end
end
