"""
    LazyTensorMappingApplication{T,R,D} <: LazyArray{T,R}

Struct for lazy application of a TensorMapping. Created using `*`.

Allows the result of a `TensorMapping` applied to a vector to be treated as an `AbstractArray`.
With a mapping `m` and a vector `v` the LazyTensorMappingApplication object can be created by `m*v`.
The actual result will be calcualted when indexing into `m*v`.
"""
struct LazyTensorMappingApplication{T,R,D, TM<:TensorMapping{T,R,D}, AA<:AbstractArray{T,D}} <: LazyArray{T,R}
    t::TM
    o::AA
end
# TODO: Do boundschecking on creation!
export LazyTensorMappingApplication

# TODO: Go through and remove unneccerary type parameters on functions

Base.:*(tm::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = LazyTensorMappingApplication(tm,o)
Base.getindex(ta::LazyTensorMappingApplication{T,R,D}, I::Vararg{Index,R}) where {T,R,D} = apply(ta.t, ta.o, I...)
Base.getindex(ta::LazyTensorMappingApplication{T,R,D}, I::Vararg{Int,R}) where {T,R,D} = apply(ta.t, ta.o, Index{Unknown}.(I)...)
Base.size(ta::LazyTensorMappingApplication) = range_size(ta.t)
# TODO: What else is needed to implement the AbstractArray interface?

# # We need the associativity to be a→b→c = a→(b→c), which is the case for '→'
Base.:*(a::TensorMapping{T,R,D}, b::TensorMapping{T,D,K}, args::Union{TensorMapping{T}, AbstractArray{T}}...) where {T,R,D,K} = foldr(*,(a,b,args...))
# # Should we overload some other infix binary opesrator?
# →(tm::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = LazyTensorMappingApplication(tm,o)
# TODO: We need to be really careful about good error messages.
# For example what happens if you try to multiply LazyTensorMappingApplication with a TensorMapping(wrong order)?

"""
    LazyTensorMappingTranspose{T,R,D} <: TensorMapping{T,D,R}

Struct for lazy transpose of a TensorMapping.

If a mapping implements the the `apply_transpose` method this allows working with
the transpose of mapping `m` by using `m'`. `m'` will work as a regular TensorMapping lazily calling
the appropriate methods of `m`.
"""
struct LazyTensorMappingTranspose{T,R,D} <: TensorMapping{T,D,R}
    tm::TensorMapping{T,R,D}
end
export LazyTensorMappingTranspose

# # TBD: Should this be implemented on a type by type basis or through a trait to provide earlier errors?
# Jonatan 2020-09-25: Is the problem that you can take the transpose of any TensorMapping even if it doesn't implement `apply_transpose`?
Base.adjoint(tm::TensorMapping) = LazyTensorMappingTranspose(tm)
Base.adjoint(tmt::LazyTensorMappingTranspose) = tmt.tm

apply(tmt::LazyTensorMappingTranspose{T,R,D}, v::AbstractArray{T,R}, I::Vararg{Index,D}) where {T,R,D} = apply_transpose(tmt.tm, v, I...)
apply_transpose(tmt::LazyTensorMappingTranspose{T,R,D}, v::AbstractArray{T,D}, I::Vararg{Index,R}) where {T,R,D} = apply(tmt.tm, v, I...)

range_size(tmt::LazyTensorMappingTranspose) = domain_size(tmt.tm)
domain_size(tmt::LazyTensorMappingTranspose) = range_size(tmt.tm)


struct LazyTensorMappingBinaryOperation{Op,T,R,D,T1<:TensorMapping{T,R,D},T2<:TensorMapping{T,R,D}} <: TensorMapping{T,D,R}
    tm1::T1
    tm2::T2

    @inline function LazyTensorMappingBinaryOperation{Op,T,R,D}(tm1::T1,tm2::T2) where {Op,T,R,D, T1<:TensorMapping{T,R,D},T2<:TensorMapping{T,R,D}}
        return new{Op,T,R,D,T1,T2}(tm1,tm2)
    end
end
# TODO: Boundschecking in constructor.

apply(tmBinOp::LazyTensorMappingBinaryOperation{:+,T,R,D}, v::AbstractArray{T,D}, I::Vararg{Index,R}) where {T,R,D} = apply(tmBinOp.tm1, v, I...) + apply(tmBinOp.tm2, v, I...)
apply(tmBinOp::LazyTensorMappingBinaryOperation{:-,T,R,D}, v::AbstractArray{T,D}, I::Vararg{Index,R}) where {T,R,D} = apply(tmBinOp.tm1, v, I...) - apply(tmBinOp.tm2, v, I...)

range_size(tmBinOp::LazyTensorMappingBinaryOperation{Op,T,R,D}) where {Op,T,R,D} = range_size(tmBinOp.tm1)
domain_size(tmBinOp::LazyTensorMappingBinaryOperation{Op,T,R,D}) where {Op,T,R,D} = domain_size(tmBinOp.tm1)

Base.:+(tm1::TensorMapping{T,R,D}, tm2::TensorMapping{T,R,D}) where {T,R,D} = LazyTensorMappingBinaryOperation{:+,T,R,D}(tm1,tm2)
Base.:-(tm1::TensorMapping{T,R,D}, tm2::TensorMapping{T,R,D}) where {T,R,D} = LazyTensorMappingBinaryOperation{:-,T,R,D}(tm1,tm2)

"""
    TensorMappingComposition{T,R,K,D}

Lazily compose two TensorMappings, so that they can be handled as a single TensorMapping.
"""
struct TensorMappingComposition{T,R,K,D, TM1<:TensorMapping{T,R,K}, TM2<:TensorMapping{T,K,D}} <: TensorMapping{T,R,D}
    t1::TM1
    t2::TM2

    @inline function TensorMappingComposition(t1::TensorMapping{T,R,K}, t2::TensorMapping{T,K,D}) where {T,R,K,D}
        @boundscheck if domain_size(t1) != range_size(t2)
            throw(DimensionMismatch("the first argument has domain size $(domain_size(t1)) while the second has range size $(range_size(t2)) "))
        end
        return new{T,R,K,D, typeof(t1), typeof(t2)}(t1,t2)
    end
    # Add check for matching sizes as a boundscheck
end
export TensorMappingComposition

range_size(tm::TensorMappingComposition) = range_size(tm.t1)
domain_size(tm::TensorMappingComposition) = domain_size(tm.t2)

function apply(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg{S,R} where S) where {T,R,K,D}
    apply(c.t1, c.t2*v, I...)
end

function apply_transpose(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,R}, I::Vararg{S,D} where S) where {T,R,K,D}
    apply_transpose(c.t2, c.t1'*v, I...)
end

Base.@propagate_inbounds Base.:∘(s::TensorMapping, t::TensorMapping) = TensorMappingComposition(s,t)

"""
    LazyLinearMap{T,R,D,...}(A, range_indicies, domain_indicies)

TensorMapping defined by the AbstractArray A. `range_indicies` and `domain_indicies` define which indicies of A should
be considerd the range and domain of the TensorMapping. Each set of indices must be ordered in ascending order.

For instance, if A is a m x n matrix, and range_size = (1,), domain_size = (2,), then the LazyLinearMap performs the
standard matrix-vector product on vectors of size n.
"""
struct LazyLinearMap{T,R,D, RD, AA<:AbstractArray{T,RD}} <: TensorMapping{T,R,D}
    A::AA
    range_indicies::NTuple{R,Int}
    domain_indicies::NTuple{D,Int}

    function LazyLinearMap(A::AA, range_indicies::NTuple{R,Int}, domain_indicies::NTuple{D,Int}) where {T,R,D, RD, AA<:AbstractArray{T,RD}}
        if !issorted(range_indicies) || !issorted(domain_indicies)
            throw(DomainError("range_indicies and domain_indicies must be sorted in ascending order"))
        end

        return new{T,R,D,RD,AA}(A,range_indicies,domain_indicies)
    end
end
export LazyLinearMap

range_size(llm::LazyLinearMap) = size(llm.A)[[llm.range_indicies...]]
domain_size(llm::LazyLinearMap) = size(llm.A)[[llm.domain_indicies...]]

function apply(llm::LazyLinearMap{T,R,D}, v::AbstractArray{T,D}, I::Vararg{Index,R}) where {T,R,D}
    view_index = ntuple(i->:,ndims(llm.A))
    for i ∈ 1:R
        view_index = Base.setindex(view_index, Int(I[i]), llm.range_indicies[i])
    end
    A_view = @view llm.A[view_index...]
    return sum(A_view.*v)
end

function apply_transpose(llm::LazyLinearMap{T,R,D}, v::AbstractArray{T,R}, I::Vararg{Index,D}) where {T,R,D}
    apply(LazyLinearMap(llm.A, llm.domain_indicies, llm.range_indicies), v, I...)
end


"""
    IdentityMapping{T,D} <: TensorMapping{T,D,D}

The lazy identity TensorMapping for a given size. Usefull for building up higher dimensional tensor mappings from lower
dimensional ones through outer products. Also used in the Implementation for InflatedTensorMapping.
"""
struct IdentityMapping{T,D} <: TensorMapping{T,D,D}
    size::NTuple{D,Int}
end
export IdentityMapping

IdentityMapping{T}(size::NTuple{D,Int}) where {T,D} = IdentityMapping{T,D}(size)

range_size(tmi::IdentityMapping) = tmi.size
domain_size(tmi::IdentityMapping) = tmi.size

apply(tmi::IdentityMapping{T,D}, v::AbstractArray{T,D}, I::Vararg{Any,D}) where {T,D} = v[I...]
apply_transpose(tmi::IdentityMapping{T,D}, v::AbstractArray{T,D}, I::Vararg{Any,D}) where {T,D} = v[I...]

