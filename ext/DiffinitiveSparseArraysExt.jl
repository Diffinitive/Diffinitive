module DiffinitiveSparseArraysExt

using Diffinitive
using Diffinitive.LazyTensors

using SparseArrays
using Tokens

"""
    sparse(t::LazyTensor)

The sparse matrix representation of `t`.

If `L` is a `LazyTensor` and `v` a tensor, then `A = sparse(L)` is constructed
so that `A*reshape(v,:) == reshape(L*v,:)`.
"""
function SparseArrays.sparse(t::LazyTensor)
    v = ArrayToken(:v, prod(domain_size(t)))

    v̄ = reshape(v,domain_size(t)...)
    tv = reshape(t*v̄, :)
    return Tokens._to_matrix(tv, prod(range_size(t)), prod(domain_size(t)))
end

end
