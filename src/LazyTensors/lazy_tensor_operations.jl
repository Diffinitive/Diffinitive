"""
    LazyTensorMappingApplication{T,R,D} <: LazyArray{T,R}

Struct for lazy application of a TensorMapping. Created using `*`.

Allows the result of a `TensorMapping` applied to a vector to be treated as an `AbstractArray`.
With a mapping `m` and a vector `v` the LazyTensorMappingApplication object can be created by `m*v`.
The actual result will be calcualted when indexing into `m*v`.
"""
struct LazyTensorMappingApplication{T,R,D} <: LazyArray{T,R}
    t::TensorMapping{T,R,D}
    o::AbstractArray{T,D}
end
# TODO: Do boundschecking on creation!
export LazyTensorMappingApplication

Base.:*(tm::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = LazyTensorMappingApplication(tm,o)
Base.getindex(ta::LazyTensorMappingApplication{T,R,D}, I::Vararg{Index,R}) where {T,R,D} = apply(ta.t, ta.o, I...)
Base.getindex(ta::LazyTensorMappingApplication{T,R,D}, I::Vararg{Int,R}) where {T,R,D} = apply(ta.t, ta.o, Index{Unknown}.(I)...)
Base.size(ta::LazyTensorMappingApplication{T,R,D}) where {T,R,D} = range_size(ta.t)
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

range_size(tmt::LazyTensorMappingTranspose{T,R,D}) where {T,R,D} = domain_size(tmt.tm)
domain_size(tmt::LazyTensorMappingTranspose{T,R,D}) where {T,R,D} = range_size(tmt.tm)


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


# TODO: Write tests and documentation for LazyTensorMappingComposition
# struct LazyTensorMappingComposition{T,R,K,D} <: TensorMapping{T,R,D}
#     t1::TensorMapping{T,R,K}
#     t2::TensorMapping{T,K,D}
# end

# Base.:∘(s::TensorMapping{T,R,K}, t::TensorMapping{T,K,D}) where {T,R,K,D} = LazyTensorMappingComposition(s,t)

# function range_size(tm::LazyTensorMappingComposition{T,R,K,D}, domain_size::NTuple{D,Integer}) where {T,R,K,D}
#     range_size(tm.t1, domain_size(tm.t2, domain_size))
# end

# function domain_size(tm::LazyTensorMappingComposition{T,R,K,D}, range_size::NTuple{R,Integer}) where {T,R,K,D}
#     domain_size(tm.t1, domain_size(tm.t2, range_size))
# end

# function apply(c::LazyTensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::NTuple{R,Int}) where {T,R,K,D}
#     apply(c.t1, LazyTensorMappingApplication(c.t2,v), I...)
# end

# function apply_transpose(c::LazyTensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::NTuple{D,Int}) where {T,R,K,D}
#     apply_transpose(c.t2, LazyTensorMappingApplication(c.t1',v), I...)
# end

# # Have i gone too crazy with the type parameters? Maybe they aren't all needed?

# export →
