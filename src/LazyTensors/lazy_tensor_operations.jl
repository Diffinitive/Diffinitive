"""
    TensorApplication{T,R,D} <: LazyArray{T,R}

Struct for lazy application of a LazyTensor. Created using `*`.

Allows the result of a `LazyTensor` applied to an array to be treated as an `AbstractArray`.
With a mapping `t` and an array `v` the TensorApplication object can be created by `t*v`.
The actual result will be calculated when indexing into `t*v`.
"""
struct TensorApplication{T, R, D, LT<:LazyTensor{R, D}, AA<:AbstractArray{<:Any, D}} <: LazyArray{T, R}
    t::LT
    v::AA

    function TensorApplication(t::LazyTensor{R, D}, v::AbstractArray{<:Any, D}) where {R, D}
        @boundscheck check_domain_size(t, size(v))
        I = ntuple(i->1, range_dim(t))
        T = typeof(apply(t, v, I...))
        return new{T, R, D, typeof(t), typeof(v)}(t,v)
    end
end

function Base.getindex(ta::TensorApplication{T, R}, I::Vararg{Any, R}) where {T, R}
    @boundscheck checkbounds(ta, Int.(I)...)
    return @inbounds apply(ta.t, ta.v, I...)
end
Base.@propagate_inbounds Base.getindex(ta::TensorApplication{T, 1} where T, I::CartesianIndex{1}) = ta[Tuple(I)...] # Would otherwise be caught in the previous method.
Base.size(ta::TensorApplication) = range_size(ta.t)


"""
    TensorNegation{R,D} <: LazyTensor{R,D}

The negation of a LazyTensor.
"""
struct TensorNegation{R, D, LT<:LazyTensor{R, D}} <: LazyTensor{R, D}
    t::LT
end

apply(tneg::TensorNegation, v, I...) = -apply(tneg.t, v, I...)

range_size(tneg::TensorNegation) = range_size(tneg.t)
domain_size(tneg::TensorNegation) = domain_size(tneg.t)

function Base.:(==)(tneg1::TensorNegation, tneg2::TensorNegation)
    return tneg1.t == tneg2.t
end

Base.adjoint(tneg::TensorNegation) = TensorNegation(adjoint(tneg.t))

"""
    TensorSum{R, D, ...} <: LazyTensor{R, D}

The lazy sum of 2 or more lazy tensors.
"""
struct TensorSum{R, D, LTT<:NTuple{N, LazyTensor{R, D}} where N} <: LazyTensor{R, D}
    ts::LTT

    function TensorSum{R, D}(ts::LTT) where {R, D, LTT<:NTuple{N, LazyTensor{R,D}} where N}
        @boundscheck check_equal_size(ts...)

        return new{R, D, LTT}(ts)
    end
end

"""
    TensorSum(ts::Vararg{LazyTensor})

The lazy sum of the tensors `ts`.
"""
function TensorSum(ts::Vararg{LazyTensor})
    R = range_dim(ts[1])
    D = domain_dim(ts[1])
    return TensorSum{R,D}(ts)
end

function apply(tsum::TensorSum{R, D}, v::AbstractArray{<:Any, D}, I::Vararg{Any, R}) where {R, D}
    vs = map(tsum.ts) do tm
        apply(tm,v,I...)
    end

    return +(vs...)
end

range_size(tsum::TensorSum) = range_size(tsum.ts[1])
domain_size(tsum::TensorSum) = domain_size(tsum.ts[1])

function Base.:(==)(tsum1::TensorSum, tsum2::TensorSum)
    return tsum1.ts == tsum2.ts
end

Base.adjoint(tsum::TensorSum) = TensorSum(map(adjoint, tsum.ts)...)


"""
    TensorComposition{R, K, D}

Lazily compose two `LazyTensor`s, so that they can be handled as a single `LazyTensor`.
"""
struct TensorComposition{R, K, D, LT1<:LazyTensor{R, K}, LT2<:LazyTensor{K, D}} <: LazyTensor{R, D}
    t1::LT1
    t2::LT2

    function TensorComposition(t1::LazyTensor{R, K}, t2::LazyTensor{K, D}) where {R, K, D}
        @boundscheck check_composable(t1,t2)
        return new{R, K, D, typeof(t1), typeof(t2)}(t1,t2)
    end
end

range_size(tm::TensorComposition) = range_size(tm.t1)
domain_size(tm::TensorComposition) = domain_size(tm.t2)

function apply(tcomp::TensorComposition{R, K, D}, v::AbstractArray{<:Any, D}, I::Vararg{Any, R}) where {R, K, D}
    apply(tcomp.t1, tcomp.t2*v, I...)
end

function Base.:(==)(tcomp1::TensorComposition, tcomp2::TensorComposition)
    return tcomp1.t1 == tcomp2.t1 && tcomp1.t2 == tcomp2.t2
end

Base.adjoint(tcomp::TensorComposition) = TensorComposition(adjoint(tcomp.t2), adjoint(tcomp.t1))

"""
    TensorComposition(t, it::IdentityTensor)
    TensorComposition(it::IdentityTensor, t)

Composes a `LazyTensor` `t` with an `IdentityTensor` `it`, by returning `t`
"""
function TensorComposition(t::LazyTensor{R, D}, it::IdentityTensor{D}) where {R, D}
    @boundscheck check_domain_size(t, range_size(it))
    return t
end


"""
    InflatedTensor{R, D} <: LazyTensor{R, D}

An inflated `LazyTensor` with dimensions added before and after its actual dimensions.
"""
struct InflatedTensor{R, D, D_before, R_middle, D_middle, D_after, LT<:LazyTensor{R_middle, D_middle}} <: LazyTensor{R, D}
    before::IdentityTensor{D_before}
    t::LT
    after::IdentityTensor{D_after}

    function InflatedTensor(before, t::LazyTensor, after)
        R_before = range_dim(before)
        R_middle = range_dim(t)
        R_after = range_dim(after)
        R = R_before + R_middle + R_after

        D_before = domain_dim(before)
        D_middle = domain_dim(t)
        D_after = domain_dim(after)
        D = D_before + D_middle + D_after
        return new{R, D, D_before, R_middle, D_middle, D_after, typeof(t)}(before, t, after)
    end
end

"""
    InflatedTensor(before, t, after)
    InflatedTensor(before, t)
    InflatedTensor(t, after)

The outer product of `before`, `t` and `after`, where `before` and `after` are `IdentityTensor`s.

If one of `before` or `after` is left out, a 0-dimensional `IdentityTensor` is used as the default value.

If `t` already is an `InflatedTensor`, `before` and `after` will be extended instead of
creating a nested `InflatedTensor`.
"""
InflatedTensor(::IdentityTensor, ::LazyTensor, ::IdentityTensor)

function InflatedTensor(before, inflt::InflatedTensor, after)
    return InflatedTensor(
        IdentityTensor(before.size...,  inflt.before.size...),
        inflt.t,
        IdentityTensor(inflt.after.size..., after.size...),
    )
end

InflatedTensor(before::IdentityTensor, t::LazyTensor) = InflatedTensor(before, t, IdentityTensor())
InflatedTensor(t::LazyTensor, after::IdentityTensor) = InflatedTensor(IdentityTensor(), t, after)
# Resolve ambiguity between the two previous methods
InflatedTensor(it1::IdentityTensor, it2::IdentityTensor) = InflatedTensor(it1, it2, IdentityTensor())

# TODO: Implement some pretty printing in terms of ⊗. E.g InflatedTensor(I(3),B,I(2)) -> I(3)⊗B⊗I(2)

function range_size(inflt::InflatedTensor)
    return concatenate_tuples(
        range_size(inflt.before),
        range_size(inflt.t),
        range_size(inflt.after),
    )
end

function domain_size(inflt::InflatedTensor)
    return concatenate_tuples(
        domain_size(inflt.before),
        domain_size(inflt.t),
        domain_size(inflt.after),
    )
end

function apply(inflt::InflatedTensor{R, D}, v::AbstractArray{<:Any, D}, I::Vararg{Any, R}) where {R, D}
    dim_before = range_dim(inflt.before)
    dim_domain = domain_dim(inflt.t)
    dim_range = range_dim(inflt.t)
    dim_after = range_dim(inflt.after)

    view_index, inner_index = split_index(dim_before, dim_domain, dim_range, dim_after, I...)

    v_inner = view(v, view_index...)
    return apply(inflt.t, v_inner, inner_index...)
end

function Base.:(==)(inflt1::InflatedTensor, inflt2::InflatedTensor)
    return inflt1.before == inflt2.before && inflt1.t == inflt2.t && inflt1.after == inflt2.after
end

Base.adjoint(inflt::InflatedTensor) = InflatedTensor(inflt.before, adjoint(inflt.t), inflt.after)


@doc raw"""
    outer_product(ts...)

Creates a `TensorComposition` for the outer product of `LazyTensors` `ts...`.
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
function outer_product end

function outer_product(t1::LazyTensor, t2::LazyTensor)
    inflt1 = InflatedTensor(t1, IdentityTensor(range_size(t2)))
    inflt2 = InflatedTensor(IdentityTensor(domain_size(t1)), t2)

    return inflt1∘inflt2
end

outer_product(ts::Vararg{LazyTensor}) = foldl(outer_product, ts)


"""
    inflate(tm::LazyTensor, sz, dir)

Inflate `tm` such that it gets the size `sz` in all directions except `dir`.
Here `sz[dir]` is ignored and replaced with the range and domains size of
`tm`.

An example of when this operation is useful is when extending a one
dimensional difference operator `D` to a 2D grid of a certain size. In that
case we could have

```julia
Dx = inflate(D, (10, 10), 1)
Dy = inflate(D, (10, 10), 2)
```
"""
function inflate(t::LazyTensor, sz, dir)
    Is = IdentityTensor.(sz)
    parts = Base.setindex(Is, t, dir)
    return foldl(⊗, parts)
end
