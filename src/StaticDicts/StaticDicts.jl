module StaticDicts

export StaticDict

# Vidar 2021-02-27
#NOTE: This type was added since ==-comparison of structs containing
# Dict (even Base.ImmutableDict) fails even though the fields satisfy
# ==-comparison. This is due to the fact that === is called for Dict-fields.
# See https://github.com/JuliaLang/julia/issues/4648. If the PR gets resolved
# we should consider removing StaticDict.
"""
    StaticDict{K,V,N}(NTuple{N,Pair{K,V}})

A simple static dictonary. Performs lookup using linear search with ==-comparison
of keys. No hashing is used.
"""
struct StaticDict{K,V,N} <: AbstractDict{K,V}
    pairs::NTuple{N,Pair{K,V}}
end

function StaticDict(pairs::Vararg{Pair})
    K = typejoin(firsttype.(pairs)...)
    V = typejoin(secondtype.(pairs)...)
    N = length(pairs)
    return StaticDict{K,V,N}(pairs)
end

function Base.get(d::StaticDict, key, default)
    for p ∈ d.pairs # TBD: Is this the best? Should we use the iterator on `d`?
        if key == p.first
            return p.second
        end
    end

    return default
end

firsttype(::Pair{T1,T2}) where {T1,T2} = T1
secondtype(::Pair{T1,T2}) where {T1,T2}  = T2

Base.iterate(d::StaticDict) = iterate(d.pairs)
Base.iterate(d::StaticDict, state) = iterate(d.pairs,state)

Base.length(d::StaticDict) = length(d.pairs)


# TODO documentation: duplicate keys not allowed atm.  will error
function Base.merge(d1::StaticDict, d2::StaticDict)
    return StaticDict(d1.pairs..., d2.pairs...)
end

end # module
