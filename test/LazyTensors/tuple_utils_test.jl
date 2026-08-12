using Test
using Diffinitive.LazyTensors
using Diffinitive.LazyTensors: TupleTable, tuple_range

@testset "split_index" begin
    @test LazyTensors.split_index(2,1,2,2, 1,2,3,4,5,6) == ((1,2,:,5,6),(3,4))
    @test LazyTensors.split_index(2,3,2,2, 1,2,3,4,5,6) == ((1,2,:,:,:,5,6),(3,4))
    @test LazyTensors.split_index(3,1,1,2, 1,2,3,4,5,6) == ((1,2,3,:,5,6),(4,))
    @test LazyTensors.split_index(3,2,1,2, 1,2,3,4,5,6) == ((1,2,3,:,:,5,6),(4,))
    @test LazyTensors.split_index(1,1,2,3, 1,2,3,4,5,6) == ((1,:,4,5,6),(2,3))
    @test LazyTensors.split_index(1,2,2,3, 1,2,3,4,5,6) == ((1,:,:,4,5,6),(2,3))

    @test LazyTensors.split_index(0,1,3,3, 1,2,3,4,5,6) == ((:,4,5,6),(1,2,3))
    @test LazyTensors.split_index(3,1,3,0, 1,2,3,4,5,6) == ((1,2,3,:),(4,5,6))

    split_index_static(::Val{dim_before}, 
                       ::Val{dim_view}, 
                       ::Val{dim_index}, 
                       ::Val{dim_after}, I...) where {dim_before,dim_view,dim_index,dim_after} = 
        LazyTensors.split_index(dim_before, dim_view, dim_index, dim_after, I...)
    @inferred split_index_static(Val(2),Val(3),Val(2),Val(2),1,2,3,2,2,4)
end

@testset "split_tuple" begin
    @testset "general" begin
        @test LazyTensors.split_tuple((),()) == ()
        @test LazyTensors.split_tuple((),(0,)) == ((),)
        @test LazyTensors.split_tuple((1,), (1,)) == tuple((1,))
        @test LazyTensors.split_tuple((1,2), (1,1)) == tuple((1,),(2,))
        @test LazyTensors.split_tuple((1,2), (0,1,1)) == tuple((),(1,),(2,))
        @test LazyTensors.split_tuple((1,2), (1,0,1)) == tuple((1,),(),(2,))
        @test LazyTensors.split_tuple((1,2), (1,1,0)) == tuple((1,),(2,),())
        @test LazyTensors.split_tuple((1,2,3,4), (2,0,1,1)) == tuple((1,2),(),(3,),(4,))

        err_msg = "length(t) must equal sum(szs)"
        @test_throws ArgumentError(err_msg) LazyTensors.split_tuple((), (2,))
        @test_throws ArgumentError(err_msg) LazyTensors.split_tuple((2,), ())
        @test_throws ArgumentError(err_msg) LazyTensors.split_tuple((1,), (2,))
        @test_throws ArgumentError(err_msg) LazyTensors.split_tuple((1,2), (1,2))
        @test_throws ArgumentError(err_msg) LazyTensors.split_tuple((1,2), (1))

        split_tuple_static(t, ::Val{SZS}) where {SZS} = LazyTensors.split_tuple(t,SZS)
        @inferred split_tuple_static((1,2,3,4,5,6), Val((3,1,2)))
        @inferred split_tuple_static((1,2,3,4),Val((3,1)))
        @inferred split_tuple_static((1,2,true,4),Val((3,1)))
        @inferred split_tuple_static((1,2,3,4,5,6),Val((3,2,1)))
        @inferred split_tuple_static((1,true,3),Val((1,1,1)))
    end
end

@testset "sizes_to_ranges" begin
    @test LazyTensors.sizes_to_ranges((1,)) == (1:1,)
    @test LazyTensors.sizes_to_ranges((2,)) == (1:2,)
    @test LazyTensors.sizes_to_ranges((2,3)) == (1:2,3:5)
    @test LazyTensors.sizes_to_ranges((3,2,4)) == (1:3,4:5,6:9)
    @test LazyTensors.sizes_to_ranges((0,2)) == (1:0,1:2)
    @test LazyTensors.sizes_to_ranges((2,0)) == (1:2,2:1)
    @test LazyTensors.sizes_to_ranges((2,0,3)) == (1:2,2:1,3:5)
end

@testset "concatenate_tuples" begin
    @test LazyTensors.concatenate_tuples(()) == ()
    @test LazyTensors.concatenate_tuples((1,)) == (1,)
    @test LazyTensors.concatenate_tuples((1,), ()) == (1,)
    @test LazyTensors.concatenate_tuples((),(1,)) == (1,)
    @test LazyTensors.concatenate_tuples((1,2,3),(4,5)) == (1,2,3,4,5)
    @test LazyTensors.concatenate_tuples((1,2,3),(4,5),(6,7)) == (1,2,3,4,5,6,7)
end

@testset "left_pad_tuple" begin
    @test LazyTensors.left_pad_tuple((1,2), 0, 2) == (1,2)
    @test LazyTensors.left_pad_tuple((1,2), 0, 3) == (0,1,2)
    @test LazyTensors.left_pad_tuple((3,2), 1, 6) == (1,1,1,1,3,2)
    err_msg = "Can't pad tuple of length 2 to 0 elements"
    @test_throws DomainError(0, err_msg) LazyTensors.left_pad_tuple((1,2), 0, 0) == (1,2)
end

@testset "right_pad_tuple" begin
    @test LazyTensors.right_pad_tuple((1,2), 0, 2) == (1,2)
    @test LazyTensors.right_pad_tuple((1,2), 0, 3) == (1,2,0)
    @test LazyTensors.right_pad_tuple((3,2), 1, 6) == (3,2,1,1,1,1)
    err_msg = "Can't pad tuple of length 2 to 0 elements"
    @test_throws DomainError(0,err_msg) LazyTensors.right_pad_tuple((1,2), 0, 0) == (1,2)
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

@testset "tuple_range()" begin
    @test tuple_range(1) == (1,)
    @test tuple_range(2) == (1,2)
    @test tuple_range(5) == (1,2,3,4,5)

    @test tuple_range(Val(1)) == (1,)
    @test tuple_range(Val(2)) == (1,2)
    @test tuple_range(Val(5)) == (1,2,3,4,5)
end
