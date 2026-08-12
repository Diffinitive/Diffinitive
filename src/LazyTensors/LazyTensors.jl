module LazyTensors

using StaticArrays

export LazyTensor
export apply
export range_dim, domain_dim
export range_size, domain_size
export check_domain_size
export check_range_size
export check_equal_size
export check_composable

export TensorApplication
export TensorComposition
export TensorNegation
export TensorSum
export outer_product
export IdentityTensor
export ZeroTensor
export ScalingTensor
export DiagonalTensor
export DenseTensor
export InflatedTensor
export VectorTensor
export VectorDotTensor
export MatrixTensor
export ⊗
export DomainSizeMismatch
export RangeSizeMismatch
export componentview
export ArrayComponentView
export dirac_delta

export LazyArray
export LazyFunctionArray
export +̃, -̃, *̃, /̃

include("lazy_tensor.jl")
include("tuple_utils.jl")
include("tensor_types.jl")
include("lazy_array.jl")
include("lazy_tensor_operations.jl")
include("lazy_tensor_operator_overloads.jl")
include("componentview.jl")

function __init__()
    if isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if exc.f != Base.:*
                return
            end

            if length(exc.args) != 2
                return
            end

            if all(arg -> arg isa LazyTensor, exc.args)
                print(io, "\nDid you mean to use `∘` to compose lazy tensors?")
            end
        end
    end
end

end # module
