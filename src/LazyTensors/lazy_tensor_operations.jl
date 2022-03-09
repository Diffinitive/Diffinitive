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

Base.getindex(ta::LazyTensorMappingApplication{T,R}, I::Vararg{Any,R}) where {T,R} = apply(ta.t, ta.o, I...)
Base.getindex(ta::LazyTensorMappingApplication{T,1}, I::CartesianIndex{1}) where {T} = apply(ta.t, ta.o, I.I...)
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

apply(tmt::LazyTensorMappingTranspose{T,R,D}, v::AbstractArray{T,R}, I::Vararg{Any,D}) where {T,R,D} = apply_transpose(tmt.tm, v, I...)
apply_transpose(tmt::LazyTensorMappingTranspose{T,R,D}, v::AbstractArray{T,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmt.tm, v, I...)

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

apply(tmBinOp::LazyTensorMappingBinaryOperation{:+,T,R,D}, v::AbstractArray{T,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmBinOp.tm1, v, I...) + apply(tmBinOp.tm2, v, I...)
apply(tmBinOp::LazyTensorMappingBinaryOperation{:-,T,R,D}, v::AbstractArray{T,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmBinOp.tm1, v, I...) - apply(tmBinOp.tm2, v, I...)

range_size(tmBinOp::LazyTensorMappingBinaryOperation) = range_size(tmBinOp.tm1)
domain_size(tmBinOp::LazyTensorMappingBinaryOperation) = domain_size(tmBinOp.tm1)

Base.:+(tm1::TensorMapping{T,R,D}, tm2::TensorMapping{T,R,D}) where {T,R,D} = LazyTensorMappingBinaryOperation{:+,T,R,D}(tm1,tm2)
Base.:-(tm1::TensorMapping{T,R,D}, tm2::TensorMapping{T,R,D}) where {T,R,D} = LazyTensorMappingBinaryOperation{:-,T,R,D}(tm1,tm2)

"""
    TensorMappingComposition{T,R,K,D}

Lazily compose two `TensorMapping`s, so that they can be handled as a single `TensorMapping`.
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

range_size(tm::TensorMappingComposition) = range_size(tm.t1)
domain_size(tm::TensorMappingComposition) = domain_size(tm.t2)

function apply(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg{Any,R}) where {T,R,K,D}
    apply(c.t1, c.t2*v, I...)
end

function apply_transpose(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,R}, I::Vararg{Any,D}) where {T,R,K,D}
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

function apply(llm::LazyLinearMap{T,R,D}, v::AbstractArray{T,D}, I::Vararg{Any,R}) where {T,R,D}
    view_index = ntuple(i->:,ndims(llm.A))
    for i ∈ 1:R
        view_index = Base.setindex(view_index, Int(I[i]), llm.range_indicies[i])
    end
    A_view = @view llm.A[view_index...]
    return sum(A_view.*v)
end

function apply_transpose(llm::LazyLinearMap{T,R,D}, v::AbstractArray{T,R}, I::Vararg{Any,D}) where {T,R,D}
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

If `tm` already is an `InflatedTensorMapping`, `before` and `after` will be extended instead of
creating a nested `InflatedTensorMapping`.
"""
InflatedTensorMapping(::IdentityMapping, ::TensorMapping, ::IdentityMapping)

function InflatedTensorMapping(before, itm::InflatedTensorMapping, after)
    return InflatedTensorMapping(
        IdentityMapping(before.size...,  itm.before.size...),
        itm.tm,
        IdentityMapping(itm.after.size..., after.size...),
    )
end

InflatedTensorMapping(before::IdentityMapping, tm::TensorMapping{T}) where T = InflatedTensorMapping(before,tm,IdentityMapping{T}())
InflatedTensorMapping(tm::TensorMapping{T}, after::IdentityMapping) where T = InflatedTensorMapping(IdentityMapping{T}(),tm,after)
# Resolve ambiguity between the two previous methods
InflatedTensorMapping(I1::IdentityMapping{T}, I2::IdentityMapping{T}) where T = InflatedTensorMapping(I1,I2,IdentityMapping{T}())

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
    dim_before = range_dim(itm.before)
    dim_domain = domain_dim(itm.tm)
    dim_range = range_dim(itm.tm)
    dim_after = range_dim(itm.after)

    view_index, inner_index = split_index(Val(dim_before), Val(dim_domain), Val(dim_range), Val(dim_after), I...)

    v_inner = view(v, view_index...)
    return apply(itm.tm, v_inner, inner_index...)
end

function apply_transpose(itm::InflatedTensorMapping{T,R,D}, v::AbstractArray{T,R}, I::Vararg{Any,D}) where {T,R,D}
    dim_before = range_dim(itm.before)
    dim_domain = domain_dim(itm.tm)
    dim_range = range_dim(itm.tm)
    dim_after = range_dim(itm.after)

    view_index, inner_index = split_index(Val(dim_before), Val(dim_range), Val(dim_domain), Val(dim_after), I...)

    v_inner = view(v, view_index...)
    return apply_transpose(itm.tm, v_inner, inner_index...)
end


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


"""
    flatten_tuple(t)

Takes a nested tuple and flattens the whole structure
"""
flatten_tuple(t::NTuple{N, Number} where N) = t
flatten_tuple(t::Tuple) = ((flatten_tuple.(t)...)...,) # simplify?
flatten_tuple(ts::Vararg) = flatten_tuple(ts)

@doc raw"""
    LazyOuterProduct(tms...)

Creates a `TensorMappingComposition` for the outerproduct of `tms...`.
This is done by separating the outer product into regular products of outer products involving only identity mappings and one non-identity mapping.

First let
```math
\begin{aligned}
A &= A_{I,J} \\
B &= B_{M,N} \\
C &= C_{P,Q} \\
\end{aligned}
```

where ``I``, ``M``, ``P`` are  multi-indexes for the ranges of ``A``, ``B``, ``C``, and ``J``, ``N``, ``Q`` are multi-indexes of the domains.

We use ``⊗`` to denote the outer product
```math
(A⊗B)_{IM,JN} = A_{I,J}B_{M,N}
```

We note that
```math
A⊗B⊗C = (A⊗B⊗C)_{IMP,JNQ} = A_{I,J}B_{M,N}C_{P,Q}
```
And that
```math
A⊗B⊗C = (A⊗I_{|M|}⊗I_{|P|})(I_{|J|}⊗B⊗I_{|P|})(I_{|J|}⊗I_{|N|}⊗C)
```
where ``|⋅|`` of a multi-index is a vector of sizes for each dimension. ``I_v`` denotes the identity tensor of size ``v[i]`` in each direction
To apply ``A⊗B⊗C`` we evaluate

```math
(A⊗B⊗C)v = [(A⊗I_{|M|}⊗I_{|P|})  [(I_{|J|}⊗B⊗I_{|P|}) [(I_{|J|}⊗I_{|N|}⊗C)v]]]
```
"""
function LazyOuterProduct end
export LazyOuterProduct

function LazyOuterProduct(tm1::TensorMapping{T}, tm2::TensorMapping{T}) where T
    itm1 = InflatedTensorMapping(tm1, IdentityMapping{T}(range_size(tm2)))
    itm2 = InflatedTensorMapping(IdentityMapping{T}(domain_size(tm1)),tm2)

    return itm1∘itm2
end

LazyOuterProduct(t1::IdentityMapping{T}, t2::IdentityMapping{T}) where T = IdentityMapping{T}(t1.size...,t2.size...)
LazyOuterProduct(t1::TensorMapping, t2::IdentityMapping) = InflatedTensorMapping(t1, t2)
LazyOuterProduct(t1::IdentityMapping, t2::TensorMapping) = InflatedTensorMapping(t1, t2)

LazyOuterProduct(tms::Vararg{TensorMapping}) = foldl(LazyOuterProduct, tms)

⊗(a::TensorMapping, b::TensorMapping) = LazyOuterProduct(a,b)
export ⊗


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
    print(io, "domain size $(domain_size(err.tm)) of TensorMapping not matching size $(err.sz)")
end
