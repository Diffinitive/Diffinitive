using Test
using Sbplib.LazyTensors

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


        split_tuple_static(t, ::Val{SZS}) where {SZS} = @inline LazyTensors.split_tuple(t,SZS)

        @inferred split_tuple_static((1,2,3,4,5,6), Val((3,1,2)))
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

@testset "flatten_tuple" begin
    @test LazyTensors.flatten_tuple((1,)) == (1,)
    @test LazyTensors.flatten_tuple((1,2,3,4,5,6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple((1,2,(3,4),5,6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple((1,2,(3,(4,5)),6)) == (1,2,3,4,5,6)
    @test LazyTensors.flatten_tuple(((1,2),(3,4),(5,),6)) == (1,2,3,4,5,6)
end

@testset "left_pad_tuple" begin
    @test LazyTensors.left_pad_tuple((1,2), 0, 2) == (1,2)
    @test LazyTensors.left_pad_tuple((1,2), 0, 3) == (0,1,2)
    @test LazyTensors.left_pad_tuple((3,2), 1, 6) == (1,1,1,1,3,2)

    @test_throws DomainError(0, "Can't pad tuple of length 2 to 0 elements") LazyTensors.left_pad_tuple((1,2), 0, 0) == (1,2)
end

@testset "right_pad_tuple" begin
    @test LazyTensors.right_pad_tuple((1,2), 0, 2) == (1,2)
    @test LazyTensors.right_pad_tuple((1,2), 0, 3) == (1,2,0)
    @test LazyTensors.right_pad_tuple((3,2), 1, 6) == (3,2,1,1,1,1)

    @test_throws DomainError(0, "Can't pad tuple of length 2 to 0 elements") LazyTensors.right_pad_tuple((1,2), 0, 0) == (1,2)
end
