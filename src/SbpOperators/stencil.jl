export CenteredStencil
export CenteredNestedStencil

struct Stencil{T,N}
    range::UnitRange
    weights::NTuple{N,T}

    function Stencil(range::UnitRange,weights::NTuple{N,T}) where {T, N}
        @assert length(range) == N
        new{T,N}(range,weights)
    end
end

"""
    Stencil(weights::NTuple; center::Int)

Create a stencil with the given weights with element `center` as the center of the stencil.
"""
function Stencil(weights...; center::Int)
    weights = promote(weights...)
    N = length(weights)
    range = (1:N) .- center

    return Stencil(range, weights)
end

Stencil{T,N}(s::Stencil{S,N}) where {T,S,N} = Stencil(s.range, T.(s.weights))
Stencil{T}(s::Stencil) where T = Stencil{T,length(s)}(s)

Base.convert(::Type{Stencil{T1,N}}, s::Stencil{T2,N}) where {T1,T2,N} = Stencil{T1,N}(s)
Base.convert(::Type{Stencil{T1}}, s::Stencil{T2,N}) where {T1,T2,N} = Stencil{T1,N}(s)

Base.promote_rule(::Type{Stencil{T1,N}}, ::Type{Stencil{T2,N}}) where {T1,T2,N} = Stencil{promote_type(T1,T2),N}

function CenteredStencil(weights...)
    if iseven(length(weights))
        throw(ArgumentError("a centered stencil must have an odd number of weights."))
    end

    r = length(weights) ÷ 2

    return Stencil(-r:r, weights)
end


"""
    scale(s::Stencil, a)

Scale the weights of the stencil `s` with `a` and return a new stencil.
"""
function scale(s::Stencil, a)
    return Stencil(s.range, a.*s.weights)
end

Base.eltype(::Stencil{T,N}) where {T,N} = T
Base.length(::Stencil{T,N}) where {T,N} = N

function flip(s::Stencil)
    range = (-s.range[2], -s.range[1])
    return Stencil(range, reverse(s.weights))
end

# Provides index into the Stencil based on offset for the root element
@inline function Base.getindex(s::Stencil, i::Int)
    @boundscheck if i ∉ s.range
        return zero(eltype(s))
    end
    return s.weights[1 + i - s.range[1]]
end

Base.@propagate_inbounds @inline function apply_stencil(s::Stencil, v::AbstractVector, i::Int)
    w = zero(eltype(v))
    @simd for k ∈ 1:length(s)
        w += s.weights[k]*v[i + s.range[k]]
    end

    return w
end

Base.@propagate_inbounds @inline function apply_stencil_backwards(s::Stencil, v::AbstractVector, i::Int)
    w = zero(eltype(v))
    @simd for k ∈ length(s):-1:1
        w += s.weights[k]*v[i - s.range[k]]
    end
    return w
end


struct NestedStencil{T,N,M}
    s::Stencil{Stencil{T,N},M}
end

# Stencil input
NestedStencil(s::Vararg{Stencil}; center) = NestedStencil(Stencil(s... ; center))
CenteredNestedStencil(s::Vararg{Stencil}) = NestedStencil(CenteredStencil(s...))

# Tuple input
function NestedStencil(weights::Vararg{NTuple{N,Any}}; center) where N
    inner_stencils = map(w -> Stencil(w...; center), weights)
    return NestedStencil(Stencil(inner_stencils... ; center))
end
function CenteredNestedStencil(weights::Vararg{NTuple{N,Any}}) where N
    inner_stencils = map(w->CenteredStencil(w...), weights)
    return CenteredNestedStencil(inner_stencils...)
end


# Conversion
function NestedStencil{T,N,M}(ns::NestedStencil{S,N,M}) where {T,S,N,M}
    return NestedStencil(Stencil{Stencil{T}}(ns.s))
end

function NestedStencil{T}(ns::NestedStencil{S,N,M}) where {T,S,N,M}
    NestedStencil{T,N,M}(ns)
end

function Base.convert(::Type{NestedStencil{T,N,M}}, s::NestedStencil{S,N,M}) where {T,S,N,M}
    return NestedStencil{T,N,M}(s)
end
Base.convert(::Type{NestedStencil{T}}, stencil) where T = NestedStencil{T}(stencil)

function Base.promote_rule(::Type{NestedStencil{T,N,M}}, ::Type{NestedStencil{S,N,M}}) where {T,S,N,M}
    return NestedStencil{promote_type(T,S),N,M}
end

Base.eltype(::NestedStencil{T}) where T = T

function scale(ns::NestedStencil, a)
    range = ns.s.range
    weights = ns.s.weights

    return NestedStencil(Stencil(range, scale.(weights,a)))
end

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
