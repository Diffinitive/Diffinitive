module StaticDicts

export StaticDict

"""
    StaticDict{K,V,N} <: AbstractDict{K,V}

A static dictionary implementing the interface for an `AbstractDict`. A
`StaticDict` is fully immutable and after creation no changes can be made.

The immutable nature means that `StaticDict` can be compared with `===`, in
constrast to regular `Dict` or `ImmutableDict` which can not. (See
<https://github.com/JuliaLang/julia/issues/4648> for details) One important
aspect of this is that `StaticDict` can be used in a struct while still
allowing the struct to be comared using the default implementation of `==` for
structs.

Lookups are done by linear search.

Duplicate keys are not allowed and an error will be thrown if they are passed
to the constructor.
"""
struct StaticDict{K,V,N} <: AbstractDict{K,V}
    pairs::NTuple{N,Pair{K,V}}

    function StaticDict{K,V}(pairs::Vararg{Pair,N}) where {K,V,N}
        if !allunique(first.(pairs))
            throw(DomainError(pairs, "keys must be unique"))
        end
        return new{K,V,N}(pairs)
    end
end

function StaticDict(pairs::Vararg{Pair})
    K = typejoin(firsttype.(pairs)...)
    V = typejoin(secondtype.(pairs)...)
    return StaticDict{K,V}(pairs...)
end

StaticDict(pairs::NTuple{N,Pair} where N) = StaticDict(pairs...)

function Base.get(d::StaticDict, key, default)
    for p ∈ d.pairs
        if key == p.first
            return p.second
        end
    end

    return default
end

Base.iterate(d::StaticDict) = iterate(d.pairs)
Base.iterate(d::StaticDict, state) = iterate(d.pairs,state)
Base.length(d::StaticDict) = length(d.pairs)


"""
    merge(d1::StaticDict, d2::StaticDict)

Merge two `StaticDict`. Repeating keys is considered and error. This may
change in a future version.
"""
function Base.merge(d1::StaticDict, d2::StaticDict)
    return StaticDict(d1.pairs..., d2.pairs...)
end


"""
    firsttype(::Pair{T1,T2})

The type of the first element in the pair.
"""
firsttype(::Pair{T1,T2}) where {T1,T2} = T1

"""
    secondtype(::Pair{T1,T2})

The type of the secondtype element in the pair.
"""
secondtype(::Pair{T1,T2}) where {T1,T2}  = T2

end # module
