"""
    split_index(dim_before, dim_view, dim_index, dim_after, I...)

Splits the multi-index `I` into two parts. One part which is expected to be
used as a view, and one which is expected to be used as an index.
E.g.
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
`sum(szs)` should equal `length(t)`.

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


NMTuple{N, M, T} = NTuple{N, NTuple{M, T}}


"""
   TupleTable{N, M, T <: NMTuple{N, M, Any}}

A static `N` row, `M` column table of types `T` built from tuples.

A TupleTable `t` is indexable, similar to matrix objects, i.e.,
`t[i,j]` returns the the table entry in position i, j. In contrast to, e.g.,
StaticArrays.SMatrix the elements of `t` can differ in type.
"""
struct TupleTable{N, M, T <: NMTuple{N, M, Any}}
    table::T
end

"""
   TupleTable(rows...)

A TupleTable with elements specified by `rows`.

"""
function TupleTable(rows...)
    if !allequal(length, rows)
        throw(DimensionMismatch("All rows must have the same length"))
    end
    TupleTable(rows)
end

"""
   TupleTable(A::Matrix)

A TupleTable{N, M} with elements from the N × M matrix `A`.
"""
function TupleTable(A::Matrix)
    N, M = size(A)

    return map(tuple_range(N)) do i
        map(tuple_range(M)) do j
            A[i,j]
        end
    end |> TupleTable
end

Base.size(::Type{<:TupleTable{N, M}}) where {N, M} = (N, M)
Base.size(t::TupleTable) = size(typeof(t))

Base.getindex(t::TupleTable, i, j) = t.table[i][j]

"""
   Base.adjoint(tt::TupleTable)

The adjoint of the TupleTable `tt`, similar to the adjoint of a matrix.
"""
function Base.adjoint(tt::TupleTable)
    N, M = size(tt)

    return map(tuple_range(M)) do j
        map(tuple_range(N)) do i
            adjoint(tt[i, j])
        end
    end |> TupleTable
end

function Base.:(==)(a::TupleTable, b::TupleTable)
    return a.table == b.table
end

function Base.:+(a::TupleTable, b::TupleTable)
    if size(a) != size(b)
        throw(DimensionMismatch("adding TupleTable objects of sizes $(size(a)) and $(size(b))"))
    end
    return map(a.table, b.table) do aᵢ, bᵢ
        aᵢ .+ bᵢ
    end |> TupleTable
end


"""
   tuple_range(n)

The range 1, ..., n as a tuple
"""
tuple_range(n) = ntuple(identity, n)
tuple_range(::Val{N}) where N = tuple_range(N)
