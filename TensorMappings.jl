module TensorMappings
# Needs a better name ImplicitTensorMappings? Get rid of "Tensor" in the name_

abstract type TensorMapping{T,R,D} end
abstract type TensorOperator{T,D} <: TensorMapping{T,D,D} end # Does this help?

range_dim(::TensorMapping{T,R,D}) where {T,R,D} = R
domain_dim(::TensorMapping{T,R,D}) where {T,R,D} = D

range_size(::TensorOperator{T,D}, domain_size::NTuple{D,Integer}) where {T,D} = domain_size
# More prciese domain_size type?

# Should be implemented by a TensorMapping
# ========================================
# apply(t::TensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T} =
# apply_transpose(t::TensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T} =
# Does it make sense that apply should work for any size of v? And the application adapts?
# Think about boundschecking!

# range_size(::TensorMapping{T,R,D}, domain_size::NTuple{D,Integer}) where {T,R,D} =
# More prciese domain_size type?
# range_size_of_transpose()???



# Allow using the ' operator:
struct TensorMappingTranspose{T,R,D} <: TensorMapping{T,D,R}
	tm::TensorMapping{T,R,D}
end

Base.adjoint(t::TensorMapping) = TensorMappingTranspose(t) # Maybe this should be implemented on a type by type basis or through a trait to provide earlier errors.
Base.adjoint(t::TensorMappingTranspose) = t.tm

apply(tm::TensorMappingTranspose{T,R,D}, v::AbstractArray{T,R}, I::Vararg) where {T,R,D} = apply_transpose(tm.tm, v, I...)
apply_transpose(tm::TensorMappingTranspose{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,D} = apply(tm.tm, v, I...)

# range_size(::TensorMappingTranspose{T,R,D}, domain_size::NTuple{}) = range_size_of_transpose???

struct TensorApplication{T,R,D} <: AbstractArray{T,R}
	t::TensorMapping{R,D}
	o::AbstractArray{T,D}
end

Base.size(ta::TensorApplication) = range_size(ta.t,size(ta.o))
## What else is needed so that we have a proper AbstractArray?

Base.getindex(tm::TensorApplication, I::Vararg) = apply(tm.t, tm.o, I...)

→(t::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = TensorApplication(t,o)
# Should we overload some other infix binary operator?
# * has the wrong parsing properties... a*b*c is parsed to (a*b)*c (through a*b*c = *(a,b,c))
# while a→b→c is parsed as a→(b→c)
# The associativity of the operators might be fixed somehow... (rfold/lfold?)
# ∘ also is an option but that has the same problem as * (but is not n-ary) (or is this best used for composition of Mappings?)

struct TensorMappingComposition{T,R,K,D} <: TensorMapping{T,R,D}
	t1::TensorMapping{T,R,K}
	t2::TensorMapping{T,K,D}
end

import Base.∘
∘(s::TensorMapping{T,R,K}, t::TensorMapping{T,K,D}) where {T,R,K,D} = TensorMappingComposition(s,t)

function apply(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,K,D}
	apply(c.t1, TensorApplication(c.t2,v), I...)
end

function apply_transpose(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,K,D}
	apply_transpose(c.t2, TensorApplication(c.t1',v), I...)
end

# Have i gone too crazy with the type parameters? Maybe they aren't all needed?


export apply
export apply_transpose
export range_dim
export domain_dim
export range_size
export →


end #module