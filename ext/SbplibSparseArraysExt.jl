module SbplibSparseArraysExt

using Sbplib
using Sbplib.LazyTensors

using SparseArrays
using Tokens

function SparseArrays.sparse(t::LazyTensor)
    v = ArrayToken(:v, prod(domain_size(t)))

    v̄ = reshape(v,domain_size(t)...)
    tv = reshape(t*v̄, :)
    return Tokens._to_matrix(tv, prod(range_size(t)), prod(domain_size(t)))
end

end
