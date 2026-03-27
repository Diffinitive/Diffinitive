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
apply_transpose(tmi::IdentityTensor{D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {D} = v[I...]


"""
    ScalingTensor{T,D} <: LazyTensor{D,D}

A lazy tensor that scales its input with `λ`.
"""
struct ScalingTensor{T,D} <: LazyTensor{D,D}
    λ::T
    size::NTuple{D,Int}
end

LazyTensors.apply(tm::ScalingTensor{<:Any,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where D = tm.λ*v[I...]
LazyTensors.apply_transpose(tm::ScalingTensor{<:Any,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where D = tm.λ*v[I...]

LazyTensors.range_size(m::ScalingTensor) = m.size
LazyTensors.domain_size(m::ScalingTensor) = m.size


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
LazyTensors.apply_transpose(tm::DiagonalTensor{D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where D = tm.diagonal[I...]*v[I...]

Base.:(==)(a::DiagonalTensor, b::DiagonalTensor) = a.diagonal == b.diagonal


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

function apply_transpose(llm::DenseTensor{R,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {R,D}
    apply(DenseTensor(llm.A, llm.domain_indicies, llm.range_indicies), v, I...)
end

NMTuple{N,M,T} = NTuple{N,NTuple{M, T}}
# Different from SMatrix because the elements can differ in type
struct TupleTable{N,M, T <: NMTuple{N,M,Any}}
    table::T
end

function TupleTable(rows...)
    if !allequal(length, rows)
        throw(DimensionMismatch("All rows must have the same length"))
    end
    TupleTable(rows)
end

function TupleTable(A::Matrix)
    N, M = size(A)

    return map(tuple_range(N)) do i
        map(tuple_range(M)) do j
            A[i,j]
        end
    end |> TupleTable
end

Base.size(::Type{<:TupleTable{N,M}}) where {N,M} = (N,M)
Base.size(t::TupleTable) = size(typeof(t))

Base.getindex(t::TupleTable, i, j) = t.table[i][j]

function Base.adjoint(tt::TupleTable)
    N, M = size(tt)

    return map(tuple_range(M)) do j
        map(tuple_range(N)) do i
            adjoint(tt[i,j])
        end
    end |> TupleTable
end

function Base.:(==)(a::TupleTable, b::TupleTable)
    return a.table == b.table
end


## "Vector of tensors ∘ scalar -> vector"
struct VectorTensor{N,R,D,NT<:NTuple{N,LazyTensor{R,D}}} <: LazyTensor{R,D}
    D::NT
    ## TODO: add constructor with tests for checking domain and range size
end

VectorTensor(Ds::Vararg{LazyTensor}) = VectorTensor(Ds)

function apply(t::VectorTensor{N,R,D}, v::AbstractArray{<:Any, D}, I::Vararg{Any,R}) where {N,R,D}
    return map(t.D) do Dᵢ
        apply(Dᵢ, v, I...)
    end |> SVector
end

Base.adjoint(t::VectorTensor) = VectorDotTensor(map(adjoint, t.D))
LazyTensors.domain_size(t::VectorTensor) = domain_size(t.D[1])
LazyTensors.range_size(t::VectorTensor) = range_size(t.D[1])

function Base.:(==)(a::VectorTensor, b::VectorTensor)
    return a.D == b.D
end

## "Vector of tensors ∘ vector -> scalar"
struct VectorDotTensor{N,R,D,NT<:NTuple{N,LazyTensor{R,D}}} <: LazyTensor{R,D}
    D::NT
    ## TODO: add constructor with tests for checking domain and range size   (allequal(domain_size), tms)
end

VectorDotTensor(Ds::Vararg{LazyTensor}) = VectorDotTensor(Ds)

function apply(t::VectorDotTensor{N,R,D}, v::AbstractArray{<:Any, D}, I::Vararg{Any,R}) where {N,R,D}
    Dᵢvᵢs = map(tuple_range(N), t.D) do i, Dᵢ
        vᵢ = componentview(v, i)
        apply(Dᵢ, vᵢ, I...)
    end

    return +(Dᵢvᵢs...)
end

Base.adjoint(t::VectorDotTensor) = VectorTensor(map(adjoint, t.D))
LazyTensors.domain_size(t::VectorDotTensor) = domain_size(t.D[1])
LazyTensors.range_size(t::VectorDotTensor) = range_size(t.D[1])


## "Matrix of tensors ∘ vector -> vector"
struct MatrixTensor{N,M,R,D,TT<:TupleTable{N,M,<:NMTuple{N,M,LazyTensor{R,D}}}} <: LazyTensor{R,D}
    D::TT # Matrix of Tensors
end

function MatrixTensor(Ds::Vararg{NTuple{N, LazyTensor} where N})
    return MatrixTensor(TupleTable(Ds))
end

function MatrixTensor(Ds::Matrix)
    return MatrixTensor(TupleTable(Ds))
end

function apply(t::MatrixTensor{N,M,R,D}, v::AbstractArray{<:Any, D}, I::Vararg{Any,R}) where {N,M,R,D}
    return map(tuple_range(N)) do i
        @inline
        Dᵢⱼvⱼs = map(tuple_range(M), t.D[i,:]) do j, Dᵢⱼ
            vⱼ = componentview(v, j)
            apply(Dᵢⱼ, vⱼ, I...)
        end

        +(Dᵢⱼvⱼs...)
    end |> SVector
end

Base.adjoint(t::MatrixTensor) = MatrixTensor(adjoint(t.D))
LazyTensors.domain_size(t::MatrixTensor) = domain_size(t.D[1,1])
LazyTensors.range_size(t::MatrixTensor) = range_size(t.D[1,1])

tuple_range(n) = ntuple(identity, n)
