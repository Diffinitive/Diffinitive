function dissipation(g::EquidistantGrid, p, direction)
    h_inv = inverse_spacing(g)[direction]

    # D = volume_operator(g,CenteredStencil(1),(CenteredStencil(1)), )
    return nothing, nothing
end

dissipation(g::EquidistantGrid{1}, p) = dissipation(g, p, 1)

function dissipation_interior_weights(p)
   if p == 0
       return (1,)
   end

   return (0, dissipation_interior_weights(p-1)...) .- (dissipation_interior_weights(p-1)..., 0)
end

function dissipation_interior_stencil(p)
    w = dissipation_interior_weights(p)
    Stencil(w..., center=midpoint(w))
end

function dissipation_transpose_interior_stencil(p)
    w = dissipation_interior_weights(p)
    Stencil(w..., center=midpoint_transpose(w))
end


midpoint(weights) = length(weights)÷2 + 1
midpoint_transpose(weights) = length(weights)+1 - midpoint(weights)

dissipation_lower_closure_size(weights) = midpoint(weights) - 1
dissipation_upper_closure_size(weights) = length(weights) - midpoint(weights)

dissipation_lower_closure_stencils(interior_weights) = ntuple(i->Stencil(interior_weights..., center=i                       ), dissipation_lower_closure_size(interior_weights))
dissipation_upper_closure_stencils(interior_weights) = ntuple(i->Stencil(interior_weights..., center=length(interior_weights)-dissipation_upper_closure_size(interior_weights)+i), dissipation_upper_closure_size(interior_weights))

dissipation_transpose_lower_closure_stencils(interior_weights) =         ntuple(i->dissipation_transpose_lower_closure_stencil(interior_weights, i), length(interior_weights))
dissipation_transpose_upper_closure_stencils(interior_weights) = reverse(ntuple(i->dissipation_transpose_upper_closure_stencil(interior_weights, i), length(interior_weights)))


function dissipation_transpose_lower_closure_stencil(interior_weights, i)
    w = ntuple(k->interior_weights[i], dissipation_lower_closure_size(interior_weights))

    for k ∈ i:-1:1
        w = (w..., interior_weights[k])
    end

    return Stencil(w..., center = i)
end

function dissipation_transpose_upper_closure_stencil(interior_weights, i)
    j = length(interior_weights)+1-i
    w = ntuple(k->interior_weights[j], dissipation_upper_closure_size(interior_weights))

    for k ∈ j:1:length(interior_weights)
        w = (interior_weights[k], w...)
    end

    return Stencil(w..., center = length(interior_weights)-midpoint(interior_weights)+1)
end
