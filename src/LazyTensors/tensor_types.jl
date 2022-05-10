"""
    IdentityTensor{T,D} <: LazyTensor{T,D,D}

The lazy identity LazyTensor for a given size. Usefull for building up higher dimensional tensor mappings from lower
dimensional ones through outer products. Also used in the Implementation for InflatedTensor.
"""
struct IdentityTensor{T,D} <: LazyTensor{T,D,D}
    size::NTuple{D,Int}
end

IdentityTensor{T}(size::NTuple{D,Int}) where {T,D} = IdentityTensor{T,D}(size)
IdentityTensor{T}(size::Vararg{Int,D}) where {T,D} = IdentityTensor{T,D}(size)
IdentityTensor(size::Vararg{Int,D}) where D = IdentityTensor{Float64,D}(size)

range_size(tmi::IdentityTensor) = tmi.size
domain_size(tmi::IdentityTensor) = tmi.size

apply(tmi::IdentityTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = v[I...]
apply_transpose(tmi::IdentityTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = v[I...]


"""
    ScalingTensor{T,D} <: LazyTensor{T,D,D}

A lazy tensor that scales its input with `λ`.
"""
struct ScalingTensor{T,D} <: LazyTensor{T,D,D}
    λ::T
    size::NTuple{D,Int}
end

LazyTensors.apply(tm::ScalingTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = tm.λ*v[I...]
LazyTensors.apply_transpose(tm::ScalingTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = tm.λ*v[I...]

LazyTensors.range_size(m::ScalingTensor) = m.size
LazyTensors.domain_size(m::ScalingTensor) = m.size


"""
    DiagonalTensor{T,D,...} <: LazyTensor{T,D,D}
    DiagonalTensor(a::AbstractArray)

A lazy tensor with diagonal `a`.
"""
struct DiagonalTensor{T,D,AT<:AbstractArray{T,D}} <: LazyTensor{T,D,D}
    diagonal::AT
end

range_size(tm::DiagonalTensor) = size(tm.diagonal)
domain_size(tm::DiagonalTensor) = size(tm.diagonal)


LazyTensors.apply(tm::DiagonalTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = tm.diagonal[I...]*v[I...]
LazyTensors.apply_transpose(tm::DiagonalTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = tm.diagonal[I...]*v[I...]


"""
    DenseTensor{T,R,D,...}(A, range_indicies, domain_indicies)

LazyTensor defined by the AbstractArray A. `range_indicies` and `domain_indicies` define which indicies of A should
be considerd the range and domain of the LazyTensor. Each set of indices must be ordered in ascending order.

For instance, if A is a m x n matrix, and range_size = (1,), domain_size = (2,), then the DenseTensor performs the
standard matrix-vector product on vectors of size n.
"""
struct DenseTensor{T,R,D, RD, AA<:AbstractArray{T,RD}} <: LazyTensor{T,R,D}
    A::AA
    range_indicies::NTuple{R,Int}
    domain_indicies::NTuple{D,Int}

    function DenseTensor(A::AA, range_indicies::NTuple{R,Int}, domain_indicies::NTuple{D,Int}) where {T,R,D, RD, AA<:AbstractArray{T,RD}}
        if !issorted(range_indicies) || !issorted(domain_indicies)
            throw(DomainError("range_indicies and domain_indicies must be sorted in ascending order"))
        end

        return new{T,R,D,RD,AA}(A,range_indicies,domain_indicies)
    end
end

range_size(llm::DenseTensor) = size(llm.A)[[llm.range_indicies...]]
domain_size(llm::DenseTensor) = size(llm.A)[[llm.domain_indicies...]]

function apply(llm::DenseTensor{T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D}
    view_index = ntuple(i->:,ndims(llm.A))
    for i ∈ 1:R
        view_index = Base.setindex(view_index, Int(I[i]), llm.range_indicies[i])
    end
    A_view = @view llm.A[view_index...]
    return sum(A_view.*v)
end

function apply_transpose(llm::DenseTensor{T,R,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {T,R,D}
    apply(DenseTensor(llm.A, llm.domain_indicies, llm.range_indicies), v, I...)
end
