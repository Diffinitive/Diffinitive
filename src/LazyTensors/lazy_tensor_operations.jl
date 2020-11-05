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

Base.getindex(ta::LazyTensorMappingApplication{T,R,D}, I::Vararg{Index,R}) where {T,R,D} = apply(ta.t, ta.o, I...)
Base.getindex(ta::LazyTensorMappingApplication{T,R,D}, I::Vararg{Int,R}) where {T,R,D} = apply(ta.t, ta.o, Index{Unknown}.(I)...)
Base.size(ta::LazyTensorMappingApplication) = range_size(ta.t)
# TODO: What else is needed to implement the AbstractArray interface?

Base.:*(a::TensorMapping, v::AbstractArray) = LazyTensorMappingApplication(a,v)
Base.:*(a::TensorMapping, b::TensorMapping) = throw(MethodError(Base.:*,(a,b)))
Base.:*(a::TensorMapping, args::Union{TensorMapping, AbstractArray}...) = foldr(*,(a,args...))

# # We need the associativity to be a→b→c = a→(b→c), which is the case for '→'
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
struct LazyTensorMappingTranspose{T,R,D, TM<:TensorMapping{T,R,D}} <: TensorMapping{T,D,R}
    tm::TM
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
        @boundscheck check_domain_size(t1, range_size(t2))
        return new{T,R,K,D, typeof(t1), typeof(t2)}(t1,t2)
    end
end
export TensorMappingComposition

function check_domain_size(tm::TensorMapping, sz)
    if domain_size(tm) != sz
        throw(SizeMismatch(tm,sz))
    end
end

struct SizeMismatch <: Exception
    tm::TensorMapping
    sz
end
export SizeMismatch

function Base.showerror(io::IO, err::SizeMismatch)
    print(io, "SizeMismatch: ")
    print(io, "attempt to apply TensorMapping with domain size $(domain_size(err.tm)) to a domain of size $(err.sz)")
end


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
IdentityMapping{T}(size::Vararg{Int,D}) where {T,D} = IdentityMapping{T,D}(size)
IdentityMapping(size::Vararg{Int,D}) where D = IdentityMapping{Float64,D}(size)

range_size(tmi::IdentityMapping) = tmi.size
domain_size(tmi::IdentityMapping) = tmi.size

apply(tmi::IdentityMapping{T,D}, v::AbstractArray{T,D}, I::Vararg{Any,D}) where {T,D} = v[I...]
apply_transpose(tmi::IdentityMapping{T,D}, v::AbstractArray{T,D}, I::Vararg{Any,D}) where {T,D} = v[I...]

"""
Base.:∘(tm, tmi)
Base.:∘(tmi, tm)

Composes a `Tensormapping` `tm` with an `IdentityMapping` `tmi`, by returning `tm`
"""
@inline function Base.:∘(tm::TensorMapping{T,R,D}, tmi::IdentityMapping{T,D}) where {T,R,D}
    @boundscheck check_domain_size(tm, range_size(tmi))
    return tm
end

@inline function Base.:∘(tmi::IdentityMapping{T,R}, tm::TensorMapping{T,R,D}) where {T,R,D}
    @boundscheck check_domain_size(tmi, range_size(tm))
    return tm
end
# Specialization for the case where tm is an IdentityMapping. Required to resolve ambiguity.
@inline function Base.:∘(tm::IdentityMapping{T,D}, tmi::IdentityMapping{T,D}) where {T,D}
    @boundscheck check_domain_size(tm, range_size(tmi))
    return tmi
end


"""
    InflatedTensorMapping{T,R,D} <: TensorMapping{T,R,D}

An inflated `TensorMapping` with dimensions added before and afer its actual dimensions.
"""
struct InflatedTensorMapping{T,R,D,D_before,R_middle,D_middle,D_after, TM<:TensorMapping{T,R_middle,D_middle}} <: TensorMapping{T,R,D}
    before::IdentityMapping{T,D_before}
    tm::TM
    after::IdentityMapping{T,D_after}

    function InflatedTensorMapping(before, tm::TensorMapping{T}, after) where T
        R_before = range_dim(before)
        R_middle = range_dim(tm)
        R_after = range_dim(after)
        R = R_before+R_middle+R_after

        D_before = domain_dim(before)
        D_middle = domain_dim(tm)
        D_after = domain_dim(after)
        D = D_before+D_middle+D_after
        return new{T,R,D,D_before,R_middle,D_middle,D_after, typeof(tm)}(before, tm, after)
    end
end
export InflatedTensorMapping
"""
    InflatedTensorMapping(before, tm, after)
    InflatedTensorMapping(before,tm)
    InflatedTensorMapping(tm,after)

The outer product of `before`, `tm` and `after`, where `before` and `after` are `IdentityMapping`s.

If one of `before` or `after` is left out, a 0-dimensional `IdentityMapping` is used as the default value.
"""
InflatedTensorMapping(::IdentityMapping, ::TensorMapping, ::IdentityMapping)
InflatedTensorMapping(before::IdentityMapping, tm::TensorMapping{T}) where T = InflatedTensorMapping(before,tm,IdentityMapping{T}())
InflatedTensorMapping(tm::TensorMapping{T}, after::IdentityMapping) where T = InflatedTensorMapping(IdentityMapping{T}(),tm,after)
# Resolve ambiguity between the two previous methods
InflatedTensorMapping(I1::IdentityMapping{T}, I2::IdentityMapping{T}) where T = InflatedTensorMapping(I1,I2,IdentityMapping{T}())

# TODO: Implement syntax and constructors for products of different combinations of InflatedTensorMapping and IdentityMapping

# TODO: Implement some pretty printing in terms of ⊗. E.g InflatedTensorMapping(I(3),B,I(2)) -> I(3)⊗B⊗I(2)

function range_size(itm::InflatedTensorMapping)
    return flatten_tuple(
        range_size(itm.before),
        range_size(itm.tm),
        range_size(itm.after),
    )
end

function domain_size(itm::InflatedTensorMapping)
    return flatten_tuple(
        domain_size(itm.before),
        domain_size(itm.tm),
        domain_size(itm.after),
    )
end

function apply(itm::InflatedTensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg{Any,R}) where {T,R,D}
    view_index, inner_index = split_index(itm, I...)

    v_inner = view(v, view_index...)
    return apply(itm.tm, v_inner, inner_index...)
end


"""
    split_index(...)

Splits the multi-index into two parts. One part for the view that the inner TensorMapping acts on, and one part for indexing the result
Eg.
```
(1,2,3,4) -> (1,:,:,4), (2,3)
```
"""
function split_index(itm::InflatedTensorMapping{T,R,D}, I::Vararg{Any,R}) where {T,R,D}
    I_before = slice_tuple(I, Val(1), Val(range_dim(itm.before)))
    I_after = slice_tuple(I, Val(R-range_dim(itm.after)+1), Val(R))

    view_index = (I_before..., ntuple((i)->:,domain_dim(itm.tm))..., I_after...)
    inner_index = slice_tuple(I, Val(range_dim(itm.before)+1), Val(R-range_dim(itm.after)))

    return (view_index, inner_index)
end

# TODO: Can this be replaced by something more elegant while still being type stable? 2020-10-21
# See:
# https://github.com/JuliaLang/julia/issues/34884
# https://github.com/JuliaLang/julia/issues/30386
"""
    slice_tuple(t, Val(l), Val(u))

Get a slice of a tuple in a type stable way.
Equivalent to t[l:u] but type stable.
"""
function slice_tuple(t,::Val{L},::Val{U}) where {L,U}
    return ntuple(i->t[i+L-1], U-L+1)
end

"""
    flatten_tuple(t)

Takes a nested tuple and flattens the whole structure
"""
flatten_tuple(t::NTuple{N, Number} where N) = t
flatten_tuple(t::Tuple) = ((flatten_tuple.(t)...)...,) # simplify?
flatten_tuple(ts::Vararg) = flatten_tuple(ts)
