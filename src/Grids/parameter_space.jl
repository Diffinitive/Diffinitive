"""
    ParameterSpace{D}

A space of parameters of dimension `D`.

Common parameter spaces are created using functions for unit sized spaces
* [`unitinterval`](@ref)
* [`unitsquare`](@ref)
* [`unitcube`](@ref)
* [`unithyperbox`](@ref)
* [`unittriangle`](@ref)
* [`unittetrahedron`](@ref)
* [`unitsimplex`](@ref)

See also: [`Interval`](@ref), [`HyperBox`](@ref),
[`Simplex`](@ref).
"""
abstract type ParameterSpace{D} end
Base.ndims(::ParameterSpace{D}) where D = D

@doc """
    in(x, S::ParameterSpace)
    ∈(x, S::ParameterSpace)

Test if the point `x` is in the parameter space `S`.
""" Base.in(x,::ParameterSpace)

"""
    Interval{T} <: ParameterSpace{1}

A `ParameterSpace` representing an interval.
"""
struct Interval{T} <: ParameterSpace{1}
    a::T
    b::T
end

"""
    Interval(a,b)

An interval with limits `a` and `b`.
"""
function Interval(a,b)
    a, b = promote(a, b)
    Interval{typeof(a)}(a,b)
end

"""
    limits(i::Interval)

The limits of the interval.
"""
limits(i::Interval) = (i.a, i.b)

boundary_identifiers(::Interval) = (LowerBoundary(), UpperBoundary())

Base.in(x, i::Interval) = i.a <= x <= i.b

"""
    unitinterval(T=Float64)

The interval ``(0,1)``.
"""
unitinterval(T=Float64) = Interval(zero(T), one(T))


"""
    HyperBox{T,D} <: ParameterSpace{D}

A `ParameterSpace` representing a hyper box.
"""
struct HyperBox{T,D} <: ParameterSpace{D}
    a::SVector{D,T}
    b::SVector{D,T}
end

"""
    HyperBox(a,b)

A `HyperBox` with lower limits `a` and upper limits `b` for each dimension.
"""
function HyperBox(a,b)
    ET = promote_type(eltype(a),eltype(b))
    T = SVector{length(a),ET}
    HyperBox(convert(T,a), convert(T,b))
end

Rectangle{T} = HyperBox{T,2}
Box{T} = HyperBox{T,3}

"""
    limits(box::HyperBox, d)

Limits of `box` along dimension `d`.
"""
limits(box::HyperBox, d) = (box.a[d], box.b[d])

"""
    limits(box::HyperBox)

The lower and upper limits of `box` as tuples.
"""
limits(box::HyperBox) = (box.a, box.b)

function boundary_identifiers(box::HyperBox)
    mapreduce(vcat, 1:ndims(box)) do d
        [
            CartesianBoundary{d, LowerBoundary}(),
            CartesianBoundary{d, UpperBoundary}(),
        ]
    end
end

function Base.in(x, box::HyperBox)
    return all(eachindex(x)) do i
        box.a[i] <= x[i] <= box.b[i]
    end
end

"""
    unitsquare(T=Float64)

The square limited by 0 and 1 in each dimension.
"""
unitsquare(T=Float64) = unithyperbox(T,2)

"""
    unitcube(T=Float64)

The cube limited by 0 and 1 in each dimension.
"""
unitcube(T=Float64) = unithyperbox(T,3)

"""
    unithyperbox(T=Float64, D)

The hypercube limited by 0 and 1 in each dimension.
"""
unithyperbox(T, D) = HyperBox((@SVector zeros(T,D)), (@SVector ones(T,D)))
unithyperbox(D) = unithyperbox(Float64,D)


"""
    Simplex{T,D,NV} <: ParameterSpace{D}

A `ParameterSpace` representing a simplex.
"""
struct Simplex{T,D,NV} <: ParameterSpace{D}
    verticies::NTuple{NV,SVector{D,T}}

    Simplex(verticies::Tuple{SVector{D,T}, Vararg{SVector{D,T},N}}) where {T,D,N} = new{T,D,N+1}(verticies)
    Simplex(::Tuple{}) = throw(ArgumentError("Must provide at least one vertex."))
end

"""
    Simplex(verticies...)

A simplex with the given verticies.
"""
function Simplex(verticies::Vararg{AbstractArray})
    ET = mapreduce(eltype,promote_type,verticies)
    T = SVector{length(verticies[1]),ET}

    return Simplex(Tuple(convert(T,v) for v ∈ verticies))
end

function Base.in(x, s::Simplex)
    v₁ = s.verticies[1]
    V = map(s.verticies) do v
        v - v₁
    end

    A = hcat(V[2:end]...) # Matrix with edge vectors as columns
    λ = A \ (x - v₁)

    λ_full = (1 - sum(λ), λ...) # Full barycentric coordinates

    return all(λᵢ -> zero(λᵢ) ≤ λᵢ ≤ one(λᵢ), λ_full)
end

"""
    verticies(s::Simplex)

Verticies of `s`.
"""
verticies(s::Simplex) = s.verticies

Triangle{T} = Simplex{T,2}
Tetrahedron{T} = Simplex{T,3}

"""
    unittriangle(T=Float64)

The simplex with verticies ``(0,0)``, ``(1,0)``, and ``(0,1)``.
"""
unittriangle(T=Float64) = unitsimplex(T,2)

"""
    unittetrahedron(T=Float64)

The simplex with verticies ``(0,0,0)``, ``(1,0,0)``, ``(0,1,0)``, and ``(0,0,1)``.
"""
unittetrahedron(T=Float64) = unitsimplex(T,3)

"""
    unitsimplex(T=Float64,D)

The unit simplex in dimension `D` with verticies ``(0,0,0,...)``, ``(1,0,0,...)``, ``(0,1,0,...)``, ``(0,0,1,...)``...
"""
function unitsimplex(T,D)
    z = @SVector zeros(T,D)
    unitelement = one(eltype(z))
    verticies = ntuple(i->setindex(z, unitelement, i), D)
    return Simplex((z,verticies...))
end
unitsimplex(D) = unitsimplex(Float64, D)
