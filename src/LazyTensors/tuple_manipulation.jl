"""
    split_index(::Val{dim_before}, ::Val{dim_view}, ::Val{dim_index}, ::Val{dim_after}, I...)

Splits the multi-index `I` into two parts. One part which is expected to be
used as a view, and one which is expected to be used as an index.
Eg.
```
split_index(Val(1),Val(3),Val(2),Val(1),(1,2,3,4)) -> (1,:,:,:,4), (2,3)
```

`dim_view` controls how many colons are in the view, and `dim_index` controls
how many elements are extracted from the middle.
`dim_before` and `dim_after` decides the length of the index parts before and after the colons in the view index.

Arguments should satisfy `length(I) == dim_before+B_domain+dim_after`.

The returned values satisfy
 * `length(view_index) == dim_before + dim_view + dim_after`
 * `length(I_middle) == dim_index`
"""
function split_index(::Val{dim_before}, ::Val{dim_view}, ::Val{dim_index}, ::Val{dim_after}, I...) where {dim_before,dim_view, dim_index,dim_after}
    I_before, I_middle, I_after = split_tuple(I, Val(dim_before), Val(dim_index))

    view_index = (I_before..., ntuple((i)->:, dim_view)..., I_after...)

    return view_index, I_middle
end
# TBD: If the nice split_tuple works, can this be cleaned up as well?

# TODO: Can this be replaced by something more elegant while still being type stable? 2020-10-21
# See:
# https://github.com/JuliaLang/julia/issues/34884
# https://github.com/JuliaLang/julia/issues/30386
"""
    slice_tuple(t, Val(l), Val(u))

Get a slice of a tuple in a type stable way.
Equivalent to `t[l:u]` but type stable.
"""
function slice_tuple(t,::Val{L},::Val{U}) where {L,U}
    return ntuple(i->t[i+L-1], U-L+1)
end

"""
    split_tuple(t::Tuple{...}, ::Val{M}) where {N,M}

Split the tuple `t` into two parts. the first part is `M` long.
E.g
```julia
split_tuple((1,2,3,4),Val(3)) -> (1,2,3), (4,)
```
"""
function split_tuple(t::NTuple{N,Any},::Val{M}) where {N,M}
    return slice_tuple(t,Val(1), Val(M)), slice_tuple(t,Val(M+1), Val(N))
end

"""
    split_tuple(t::Tuple{...},::Val{M},::Val{K}) where {N,M,K}

Same as `split_tuple(t::NTuple{N},::Val{M})` but splits the tuple in three parts. With the first
two parts having lenght `M` and `K`.
"""
function split_tuple(t::NTuple{N,Any},::Val{M},::Val{K}) where {N,M,K}
    p1, tail = split_tuple(t, Val(M))
    p2, p3 = split_tuple(tail, Val(K))
    return p1,p2,p3
end

# TBD Are the above defs even needed? Can the below one be used without problems?

"""
    split_tuple(t, szs)

Split the tuple `t` into a set of tuples of the sizes given in `szs`.
`sum(szs)` should equal `lenght(t)`.

E.g
```julia
split_tuple((1,2,3,4,5,6), (3,1,2)) -> (1,2,3),(4,),(5,6)
```
"""
function split_tuple(t, szs)
    if length(t) != sum(szs; init=0)
        throw(ArgumentError("length(t) must equal sum(szs)"))
    end

    rs = sizes_to_ranges(szs)
    return map(r->t[r], rs)
end

function sizes_to_ranges(szs)
    cum_szs = cumsum((0, szs...))
    return ntuple(i->cum_szs[i]+1:cum_szs[i+1], length(szs))
end

concatenate_tuples(t::Tuple,ts::Vararg{Tuple}) = (t..., concatenate_tuples(ts...)...)
concatenate_tuples(t::Tuple) = t


"""
    left_pad_tuple(t, val, N)

Left pad the tuple `t` to length `N` using the value `val`.
"""
function left_pad_tuple(t, val, N)
    if N < length(t)
        throw(DomainError(N, "Can't pad tuple of length $(length(t)) to $N elements"))
    end

    padding = ntuple(i->val, N-length(t))
    return (padding..., t...)
end

"""
    right_pad_tuple(t, val, N)

Right pad the tuple `t` to length `N` using the value `val`.
"""
function right_pad_tuple(t, val, N)
    if N < length(t)
        throw(DomainError(N, "Can't pad tuple of length $(length(t)) to $N elements"))
    end

    padding = ntuple(i->val, N-length(t))
    return (t..., padding...)
end

