export CenteredStencil
export CenteredNestedStencil

struct Stencil{T,N}
    range::Tuple{Int,Int}
    weights::NTuple{N,T}

    function Stencil(range::Tuple{Int,Int},weights::NTuple{N,T}) where {T, N}
        @assert range[2]-range[1]+1 == N
        new{T,N}(range,weights)
    end
end

"""
    Stencil(weights::NTuple; center::Int)

Create a stencil with the given weights with element `center` as the center of the stencil.
"""
function Stencil(weights::Vararg{T}; center::Int) where T # Type parameter T makes sure the weights are valid for the Stencil constuctors and throws an earlier, more readable, error
    N = length(weights)
    range = (1, N) .- center

    return Stencil(range, weights)
end

function Stencil{T}(s::Stencil) where T
    return Stencil(s.range, T.(s.weights))
end

Base.convert(::Type{Stencil{T}}, stencil) where T = Stencil{T}(stencil)

function CenteredStencil(weights::Vararg{T}) where T
    if iseven(length(weights))
        throw(ArgumentError("a centered stencil must have an odd number of weights."))
    end

    r = length(weights) ÷ 2

    return Stencil((-r, r), weights)
end


"""
    scale(s::Stencil, a)

Scale the weights of the stencil `s` with `a` and return a new stencil.
"""
function scale(s::Stencil, a)
    return Stencil(s.range, a.*s.weights)
end

Base.eltype(::Stencil{T}) where T = T

function flip(s::Stencil)
    range = (-s.range[2], -s.range[1])
    return Stencil(range, reverse(s.weights))
end

# Provides index into the Stencil based on offset for the root element
@inline function Base.getindex(s::Stencil, i::Int)
    @boundscheck if i < s.range[1] || s.range[2] < i
        return zero(eltype(s))
    end
    return s.weights[1 + i - s.range[1]]
end

Base.@propagate_inbounds @inline function apply_stencil(s::Stencil{T,N}, v::AbstractVector, i::Int) where {T,N}
    w = s.weights[1]*v[i + s.range[1]]
    @simd for k ∈ 2:N
        w += s.weights[k]*v[i + s.range[1] + k-1]
    end
    return w
end

Base.@propagate_inbounds @inline function apply_stencil_backwards(s::Stencil{T,N}, v::AbstractVector, i::Int) where {T,N}
    w = s.weights[N]*v[i - s.range[2]]
    @simd for k ∈ N-1:-1:1
        w += s.weights[k]*v[i - s.range[1] - k + 1]
    end
    return w
end


struct NestedStencil{T,N,M}
    s::Stencil{Stencil{T,N},M}
end

# The exessive use of type parameters for the following constructors are to catch errors earlier with clearer error messages

# Stencil input
NestedStencil(s::Vararg{Stencil{T,N}}; center) where {T,N} = NestedStencil(Stencil(s... ; center))
CenteredNestedStencil(s::Vararg{Stencil{T,N}}) where {T,N} = NestedStencil(CenteredStencil(s...))

# Tuple input
function NestedStencil(weights::Vararg{NTuple{N,T}}; center) where {T,N}
    inner_stencils = map(w -> Stencil(w...; center), weights)
    return NestedStencil(Stencil(inner_stencils... ; center))
end
function CenteredNestedStencil(weights::Vararg{NTuple{N,T}}) where {T,N}
    inner_stencils = map(w->CenteredStencil(w...), weights)
    return CenteredNestedStencil(inner_stencils...)
end

Base.eltype(::NestedStencil{T}) where T = T

function flip(ns::NestedStencil)
    s_flip = flip(ns.s)
    return NestedStencil(Stencil(s_flip.range, flip.(s_flip.weights)))
end

Base.getindex(ns::NestedStencil, i::Int) = ns.s[i]

"Apply inner stencils to `c` and get a concrete stencil"
Base.@propagate_inbounds function apply_inner_stencils(ns::NestedStencil, c::AbstractVector, i::Int)
    weights = apply_stencil.(ns.s.weights, Ref(c), i)
    return Stencil(ns.s.range, weights)
end

"Apply the whole nested stencil"
Base.@propagate_inbounds function apply_stencil(ns::NestedStencil, c::AbstractVector, v::AbstractVector, i::Int)
    s = apply_inner_stencils(ns,c,i)
    return apply_stencil(s, v, i)
end

"Apply inner stencils backwards to `c` and get a concrete stencil"
Base.@propagate_inbounds @inline function apply_inner_stencils_backwards(ns::NestedStencil, c::AbstractVector, i::Int)
    weights = apply_stencil_backwards.(ns.s.weights, Ref(c), i)
    return Stencil(ns.s.range, weights)
end

"Apply the whole nested stencil backwards"
Base.@propagate_inbounds @inline function apply_stencil_backwards(ns::NestedStencil, c::AbstractVector, v::AbstractVector, i::Int)
    s = apply_inner_stencils_backwards(ns,c,i)
    return apply_stencil_backwards(s, v, i)
end
