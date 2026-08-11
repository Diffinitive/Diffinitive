"""
    IdentityTensor{D} <: LazyTensor{D,D}

The lazy identity LazyTensor for a given size. Useful for building up higher dimensional tensor mappings from lower
dimensional ones through outer products. Also used in the Implementation for InflatedTensor.
"""
struct IdentityTensor{D} <: LazyTensor{D,D}
    size::NTuple{D,Int}
end

IdentityTensor(size::Vararg{Int,D}) where D = IdentityTensor{D}(size)

range_size(tmi::IdentityTensor) = tmi.size
domain_size(tmi::IdentityTensor) = tmi.size

apply(tmi::IdentityTensor{D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {D} = v[I...]

Base.adjoint(t::IdentityTensor) = t

"""
    ZeroTensor{R,D} <: LazyTensor{R,D}

A lazy zero tensor which returns zero when applied to anything.
It provides implementations of `+` and `∘` which short circuits to simplified expressions of the results.
"""
struct ZeroTensor{R,D} <: LazyTensor{R,D}
    range_size::NTuple{R,Int}
    domain_size::NTuple{D,Int}
end

"""
    ZeroTensor(range_size, domain_size)

A lazy zero tensor with the given range and domain size.
"""
ZeroTensor(::Tuple, ::Tuple)


"""
    ZeroTensor(sz::Vararg{Int})

A lazy representation of the zero operator with range size and domain size both equal to `sz`.
"""
ZeroTensor(size::Vararg{Int}) = ZeroTensor(size)

"""
    ZeroTensor(sz::NTuple{N, Int} where N)

A lazy representation of the zero operator with range size and domain size both equal to `sz`.
"""
ZeroTensor(size::NTuple{N, Int} where N) = ZeroTensor(size, size)

Base.zero(t::LazyTensor) = ZeroTensor(range_size(t), domain_size(t))

range_size(t::ZeroTensor) = t.range_size
domain_size(t::ZeroTensor) = t.domain_size

function apply(t::ZeroTensor{R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any, R}) where {R,D}
    return zero(eltype(v))
end

Base.adjoint(t::ZeroTensor) = ZeroTensor(domain_size(t), range_size(t))


"""
    ScalingTensor{T,D} <: LazyTensor{D,D}

A lazy tensor that scales its input with `λ`.
"""
struct ScalingTensor{T,D} <: LazyTensor{D,D}
    λ::T
    size::NTuple{D,Int}
end

LazyTensors.apply(tm::ScalingTensor{<:Any,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where D = tm.λ*v[I...]

LazyTensors.range_size(m::ScalingTensor) = m.size
LazyTensors.domain_size(m::ScalingTensor) = m.size

function Base.:(==)(a::ScalingTensor, b::ScalingTensor)
    return a.λ == b.λ && a.size == b.size
end

Base.adjoint(t::ScalingTensor) = ScalingTensor(conj(t.λ), t.size)


"""
    DiagonalTensor{D, ...} <: LazyTensor{D,D}
    DiagonalTensor(a::AbstractArray)

A lazy tensor with diagonal `a`.
"""
struct DiagonalTensor{D,AT<:AbstractArray{<:Any,D}} <: LazyTensor{D,D}
    diagonal::AT
end

range_size(tm::DiagonalTensor) = size(tm.diagonal)
domain_size(tm::DiagonalTensor) = size(tm.diagonal)

LazyTensors.apply(tm::DiagonalTensor{D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where D = tm.diagonal[I...]*v[I...]

Base.:(==)(a::DiagonalTensor, b::DiagonalTensor) = a.diagonal == b.diagonal

Base.adjoint(t::DiagonalTensor) = DiagonalTensor(conj(t.diagonal))


"""
    DenseTensor{R,D,...}(A, range_indicies, domain_indicies) <: LazyTensor{R,D}

LazyTensor defined by the AbstractArray A. `range_indicies` and `domain_indicies` define which indices of A should
be considered the range and domain of the LazyTensor. Each set of indices must be ordered in ascending order.

For instance, if A is a m x n matrix, and range_size = (1,), domain_size = (2,), then the DenseTensor performs the
standard matrix-vector product on vectors of size n.
"""
struct DenseTensor{R, D, RD, AA<:AbstractArray{<:Any,RD}} <: LazyTensor{R,D}
    A::AA
    range_indicies::NTuple{R,Int}
    domain_indicies::NTuple{D,Int}

    function DenseTensor(A::AA, range_indicies::NTuple{R,Int}, domain_indicies::NTuple{D,Int}) where {R,D, RD, AA<:AbstractArray{<:Any,RD}}
        if !issorted(range_indicies) || !issorted(domain_indicies)
            throw(DomainError("range_indicies and domain_indicies must be sorted in ascending order"))
        end

        return new{R,D,RD,AA}(A,range_indicies,domain_indicies)
    end
end

range_size(llm::DenseTensor) = size(llm.A)[[llm.range_indicies...]]
domain_size(llm::DenseTensor) = size(llm.A)[[llm.domain_indicies...]]

function apply(llm::DenseTensor{R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {R,D}
    view_index = ntuple(i->:,ndims(llm.A))
    for i ∈ 1:R
        view_index = Base.setindex(view_index, Int(I[i]), llm.range_indicies[i])
    end
    A_view = @view llm.A[view_index...]
    return sum(A_view.*v)
end

function Base.:(==)(a::DenseTensor, b::DenseTensor)
    return a.A == b.A && a.range_indicies == b.range_indicies && a.domain_indicies == b.domain_indicies
end

Base.adjoint(t::DenseTensor) = DenseTensor(conj(t.A), t.domain_indicies, t.range_indicies)


# TODO: Move TupleTable somewhere else? 
# Perhaps we should create e.g. tuple_utils.jl which includes tuple_manipulation.jl
# TupleTable and tuple_range.
NMTuple{N, M, T} = NTuple{N, NTuple{M, T}}


"""
   TupleTable{N, M, T <: NMTuple{N, M, Any}}

A static `N` row, `M` column table of types `T` built from tuples.

A tuple table `t::TupleTable` is indexable, similar to matrix objects, i.e.,
t[i,j] returns the the table entry in position i, j. In contrast to, e.g., 
`StaticArrays.SMatrix` the elements of `t` can differ in type.
"""
struct TupleTable{N, M, T <: NMTuple{N, M, Any}}
    table::T
end


"""
   TupleTable(rows...)

A `TupleTable` with elements specified by `rows`.

"""
function TupleTable(rows...)
    if !allequal(length, rows)
        throw(DimensionMismatch("All rows must have the same length"))
    end
    TupleTable(rows)
end

"""
   TupleTable(A::Matrix)

A `TupleTable{N,M}` with elements from the N×M matrix `A`.
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

The adjoint of the `TupleTable` `tt`, similar to the adjoint of a matrix.
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
    VectorTensor{N,R,D} <: LazyTensor{R,D}

A vector of LazyTensors which when applied to an AbstractArray gives a
TensorApplication with vector elements.
"""
struct VectorTensor{N, R, D, NT <: NTuple{N, LazyTensor{R, D}}} <: LazyTensor{R, D}
    ts::NT
    function VectorTensor{N, R, D}(ts::NT) where {N, R, D, NT <: NTuple{N, LazyTensor{R, D}}}
        check_equal_size(ts...)
        return new{N, R, D, NT}(ts)
    end
end


"""
    VectorTensor(ts::NTuple{N, LazyTensor}) where N

A VectorTensor with elements `ts`.
"""
function VectorTensor(ts::NTuple{N, LazyTensor}) where N
    R = range_dim(ts[1])
    D = domain_dim(ts[1])
    return VectorTensor{N, R, D}(ts)
end

"""
    VectorTensor(ts::Vararg{LazyTensor})

A VectorTensor with elements `ts`.
"""
VectorTensor(ts::Vararg{LazyTensor}) = VectorTensor(ts)


"""
    VectorTensor(f, n}

A VectorTensor of `n` LazyTensors with elements determined by `f(i)` for ``i = 1...n``.
"""
function VectorTensor(f, n)
    return VectorTensor(map(f, tuple_range(n)))
end

function apply(vt::VectorTensor{N, R, D}, v::AbstractArray{<:Any, D}, I::Vararg{Any, R}) where {N, R, D}
    return map(vt.ts) do tᵢ
        apply(tᵢ, v, I...)
    end |> SVector
end

Base.adjoint(t::VectorTensor) = VectorDotTensor(map(adjoint, t.ts))
Base.length(::VectorTensor{N}) where N = N
LazyTensors.domain_size(t::VectorTensor) = domain_size(t.ts[1])
LazyTensors.range_size(t::VectorTensor) = range_size(t.ts[1])

function Base.:(==)(a::VectorTensor, b::VectorTensor)
    return a.ts == b.ts
end

"""
    VectorDotTensor{N,R,D} <: LazyTensor{R,D}

A vector of LazyTensors `ts` which when applied to an AbstractArray of vectors `a` gives a
TensorApplication where each element `I` is the sum Σᵢ (ts[i]*a[:][i])[I].
"""
struct VectorDotTensor{N, R, D, NT <: NTuple{N, LazyTensor{R, D}}} <: LazyTensor{R, D}
    ts::NT
    function VectorDotTensor{N, R, D}(ts::NT) where {N, R, D, NT <: NTuple{N, LazyTensor{R, D}}}
        check_equal_size(ts...)
        return new{N, R, D, NT}(ts)
    end
end


"""
    VectorDotTensor(ts::NTuple{N, LazyTensor}) where N

A VectorDotTensor with elements `ts`.
"""
function VectorDotTensor(ts::NTuple{N, LazyTensor}) where N
    R = range_dim(ts[1])
    D = domain_dim(ts[1])
    return VectorDotTensor{N, R, D}(ts)
end

"""
    VectorDotTensor(ts::Vararg{LazyTensor})

A VectorDotTensor with elements `ts`.
"""
VectorDotTensor(ts::Vararg{LazyTensor}) = VectorDotTensor(ts)


"""
    VectorDotTensor(f, n}

A VectorDotTensor of `n` LazyTensors with elements determined by `f(i)` for ``i = 1...n``.
"""
function VectorDotTensor(f, n)
    return VectorDotTensor(map(f, tuple_range(n)))
end

function apply(vdt::VectorDotTensor{N,R,D}, v::AbstractArray{<:Any, D}, I::Vararg{Any,R}) where {N,R,D}
    tᵢvᵢs = map(tuple_range(N), vdt.ts) do i, tᵢ
        vᵢ = componentview(v, i)
        apply(tᵢ, vᵢ, I...)
    end

    return +(tᵢvᵢs...)
end

Base.adjoint(vdt::VectorDotTensor) = VectorTensor(map(adjoint, vdt.ts))
Base.length(::VectorDotTensor{N}) where N = N
LazyTensors.domain_size(vdt::VectorDotTensor) = domain_size(vdt.ts[1])
LazyTensors.range_size(vdt::VectorDotTensor) = range_size(vdt.ts[1])

function Base.:(==)(a::VectorDotTensor, b::VectorDotTensor)
    return a.ts == b.ts
end

"""
    MatrixTensor{N, M, R, D} <: LazyTensor{R, D}

A matrix of LazyTensors `ts` which when applied to an AbstractArray of vectors `a` gives a
TensorApplication `ta` such that  ta[I][i] equals the sum Σⱼ (ts[i,j]*a[:][j])[I].
"""
struct MatrixTensor{N, M, R, D, TT <: TupleTable{N, M, <:NMTuple{N, M, LazyTensor{R, D}}}} <: LazyTensor{R, D}
    ts::TT # Matrix of Tensors
    function MatrixTensor{N, M, R, D}(ts::TT) where {N, M, R, D, TT <: TupleTable{N, M, <:NMTuple{N, M, LazyTensor{R, D}}}}
        check_equal_size((ts.table...)...)
        new{N, M, R, D, TT}(ts)
    end
end

"""
    MatrixTensor(ts::TupleTable)

A MatrixTensor with elements `ts`.
"""
function MatrixTensor(ts::TupleTable{N, M, <:NMTuple{N, M, LazyTensor}}) where {N, M}
    R = range_dim(ts[1,1])
    D = domain_dim(ts[1,1])
    return MatrixTensor{N, M, R, D}(ts)
end

"""
    MatrixTensor(ts::NTuple{N, NTuple{M, LazyTensor}} where {N,M})

A MatrixTensor with elements `ts`.
"""
MatrixTensor(ts::NTuple{N, NTuple{M, LazyTensor}} where {N, M}) = MatrixTensor(TupleTable(ts))

"""
    MatrixTensor(ts::Vararg{NTuple{N, LazyTensor} where N)

A MatrixTensor with elements `ts`.
"""
MatrixTensor(ts::Vararg{NTuple{N, LazyTensor} where N}) = MatrixTensor(ts)

"""
    MatrixTensor(ts::Matrix}

A MatrixTensor with elements `ts`.
"""
function MatrixTensor(ts::Matrix)
    return MatrixTensor(TupleTable(ts))
end

"""
    MatrixTensor(f, n, m}

An MatrixTensor of n×m LazyTensors with elements determined by `f(i,j)` for ``i = 1...n`` and ``j = 1...m``.
"""
function MatrixTensor(f, n, m)
    return map(tuple_range(n)) do i
        map(tuple_range(m)) do j
            f(i,j)
        end
    end |> MatrixTensor
end

MatrixTensor(::Tuple{}) = throw(ArgumentError("All dimensions of a MatrixTensor must be larger than 1"))
MatrixTensor(::NTuple{N, Tuple{}} where N) = throw(ArgumentError("The number of columns of a MatrixTensor must be larger than 1"))

function apply(mt::MatrixTensor{N,M,R,D}, v::AbstractArray{<:Any, D}, I::Vararg{Any,R}) where {N,M,R,D}
    return map(tuple_range(N)) do i
        @inline
        tᵢⱼvⱼs = map(tuple_range(M), mt.ts[i,:]) do j, tᵢⱼ
            vⱼ = componentview(v, j)
            apply(tᵢⱼ, vⱼ, I...)
        end

        +(tᵢⱼvⱼs...)
    end |> SVector
end

Base.adjoint(mt::MatrixTensor) = MatrixTensor(adjoint(mt.ts))
LazyTensors.domain_size(mt::MatrixTensor) = domain_size(mt.ts[1,1])
LazyTensors.range_size(mt::MatrixTensor) = range_size(mt.ts[1,1])
Base.size(mt::MatrixTensor) = size(mt.ts)

function Base.:(==)(a::MatrixTensor, b::MatrixTensor)
    return a.ts == b.ts
end

"""
   tuple_range(n)

The range 1, ..., n as a tuple
"""
tuple_range(n) = ntuple(identity, n)
