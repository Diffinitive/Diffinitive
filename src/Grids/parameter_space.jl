"""
    ParameterSpace{D}

A space of parameters of dimension `D`. Used with `Chart` to indicate which
parameters are valid for that chart.

Common parameter spaces are created using the functions unit sized spaces
* `unitinterval`
* `unitrectangle`
* `unitbox`
* `unittriangle`
* `unittetrahedron`
* `unithyperbox`
* `unitsimplex`

See also: [`Interval`](@ref), [`Rectangle`](@ref), [`Box`](@ref),
[`Triangle`](@ref), [`Tetrahedron`](@ref), [`HyperBox`](@ref),
[`Simplex`](@ref),
"""
abstract type ParameterSpace{D} end
Base.ndims(::ParameterSpace{D}) where D = D

struct Interval{T} <: ParameterSpace{1}
    a::T
    b::T

    function Interval(a,b)
        a, b = promote(a, b)
        new{typeof(a)}(a,b)
    end
end

limits(i::Interval) = (i.a, i.b)

unitinterval(T=Float64) = Interval(zero(T), one(T))


struct HyperBox{T,D} <: ParameterSpace{D}
    a::SVector{D,T}
    b::SVector{D,T}
end

function HyperBox(a,b)
    ET = promote_type(eltype(a),eltype(b))
    T = SVector{length(a),ET}
    HyperBox(convert(T,a), convert(T,b))
end

Rectangle{T} = HyperBox{T,2}
Box{T} = HyperBox{T,3}

limits(box::HyperBox, d) = (box.a[d], box.b[d])
limits(box::HyperBox) = (box.a, box.b)

unitsquare(T=Float64) = unithyperbox(T,2)
unitcube(T=Float64) = unithyperbox(T,3)
unithyperbox(T, D) = HyperBox((@SVector zeros(T,D)), (@SVector ones(T,D)))
unithyperbox(D) = unithyperbox(Float64,D)


struct Simplex{T,D,NV} <: ParameterSpace{D}
    verticies::NTuple{NV,SVector{D,T}}

    Simplex(verticies::Tuple{SVector{D,T}, Vararg{SVector{D,T},N}}) where {T,D,N} = new{T,D,N+1}(verticies)
    Simplex(::Tuple{}) = throw(ArgumentError("Must provide at least one vertex."))
end

function Simplex(verticies::Vararg{AbstractArray})
    ET = mapreduce(eltype,promote_type,verticies)
    T = SVector{length(verticies[1]),ET}

    return Simplex(Tuple(convert(T,v) for v ∈ verticies))
end

verticies(s::Simplex) = s.verticies

Triangle{T} = Simplex{T,2}
Tetrahedron{T} = Simplex{T,3}

unittriangle(T=Float64) = unitsimplex(T,2)
unittetrahedron(T=Float64) = unitsimplex(T,3)
function unitsimplex(T,D)
    z = @SVector zeros(T,D)
    unitelement = one(eltype(z))
    verticies = ntuple(i->setindex(z, unitelement, i), D)
    return Simplex((z,verticies...))
end
unitsimplex(D) = unitsimplex(Float64, D)
