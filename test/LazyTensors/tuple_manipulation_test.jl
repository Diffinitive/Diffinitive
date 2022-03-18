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
