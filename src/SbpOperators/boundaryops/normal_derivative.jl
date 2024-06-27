"""
    normal_derivative(g, stencil_set::StencilSet, boundary)
    normal_derivative(g::TensorGrid, stencil_set::StencilSet, boundary::TensorGridBoundary)
    normal_derivative(g::EquidistantGrid, stencil_set::StencilSet, boundary)

Creates the normal derivative boundary operator `d` as a `LazyTensor`

`d` computes the normal derivative at `boundary` of a grid function on `g` using the
'd1' stencil in `stencil_set`. `d'` is the prolongation of the normal
derivative of a grid function to the whole of `g` using the same stencil. On a
one-dimensional grid, `d` is a `BoundaryOperator`. On a multi-dimensional
grid, `d` is the inflation of a `BoundaryOperator`.

See also: [`BoundaryOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
function normal_derivative end


function normal_derivative(g::TensorGrid, stencil_set::StencilSet, boundary::TensorGridBoundary)
    op = normal_derivative(g.grids[grid_id(boundary)], stencil_set, boundary_id(boundary))
    return LazyTensors.inflate(op, size(g), grid_id(boundary))
end

function normal_derivative(g::EquidistantGrid, stencil_set::StencilSet, boundary)
    closure_stencil = parse_stencil(stencil_set["d1"]["closure"])
    h_inv = inverse_spacing(g)

    scaled_stencil = scale(closure_stencil,h_inv)
    return BoundaryOperator(g, scaled_stencil, boundary)
end

function normal_derivative(g::MappedGrid, stencil_set::StencilSet, boundary)
    b_indices = boundary_indices(logicalgrid(g), boundary)

    k = grid_id(boundary)


    # Compute the weights for the logival derivatives
    g⁻¹ = geometric_tensor_inverse(g)
    α = map(CartesianIndices(g⁻¹)[b_indices...]) do I # TODO: Fix iterator here
        gᵏⁱ = g⁻¹[I][k,:]
        gᵏᵏ = g⁻¹[I][k,k]

        gᵏⁱ./sqrt(gᵏᵏ)
    end


    mapreduce(+,1:ndims(g)) do i
        if i == k
            ∂ξᵢ = normal_derivative(logicalgrid(g), stencil_set, boundary)
        else
            ∂ξᵢ = first_derivative(logicalgrid(g), stencil_set, i)
        end

        αᵢ = componentview(α,i)
        DiagonalTensor(αᵢ) ∘ ∂ξᵢ
    end
end
