"""
    split_index(dim_before, dim_view, dim_index, dim_after, I...)

Splits the multi-index `I` into two parts. One part which is expected to be
used as a view, and one which is expected to be used as an index.
Eg.
```julia-repl
julia> LazyTensors.split_index(1, 3, 2, 1, (1,2,3,4)...)
((1, Colon(), Colon(), Colon(), 4), (2, 3))
```

`dim_view` controls how many colons are in the view, and `dim_index` controls
how many elements are extracted from the middle.
`dim_before` and `dim_after` decides the length of the index parts before and after the colons in the view index.

Arguments should satisfy `length(I) == dim_before+B_domain+dim_after`.

The returned values satisfy
 * `length(view_index) == dim_before + dim_view + dim_after`
 * `length(I_middle) == dim_index`
"""
function split_index(dim_before, dim_view, dim_index, dim_after, I...)
    @inline
    I_before, I_middle, I_after = split_tuple(I, (dim_before, dim_index, dim_after))

    view_index = (I_before..., ntuple((i)->:, dim_view)..., I_after...)

    return view_index, I_middle
end


"""
    split_tuple(t, szs)

Split the tuple `t` into a set of tuples of the sizes given in `szs`.
`sum(szs)` should equal `lenght(t)`.

E.g
```julia-repl
julia> LazyTensors.split_tuple((1,2,3,4,5,6), (3,1,2))
((1, 2, 3), (4,), (5, 6))
```
"""
function split_tuple(t, szs)
    @inline
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


"""
    concatenate_tuples(t...)

Concatenate tuples.
"""
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

