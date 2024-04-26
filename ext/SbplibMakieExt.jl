module SbplibMakieExt

using Sbplib.Grids
using Makie


function verticies_and_faces_and_values(g::Grid{<:Any,2}, gf::AbstractArray{<:Any, 2})
    ps = map(Tuple, g)[:]
    values = gf[:]

    N = length(ps)

    faces = Vector{NTuple{3,Int}}()

    n,m = size(g)
    Li = LinearIndices((1:n, 1:m))
    for i ∈ 1:n-1, j = 1:m-1

        # Add point in the middle of the patch to preserve symmetries
        push!(ps, Tuple((g[i,j] + g[i+1,j] + g[i+1,j+1] + g[i,j+1])/4))
        push!(values, (gf[i,j] + gf[i+1,j] + gf[i+1,j+1] + gf[i,j+1])/4)

        push!(faces, (Li[i,j],     Li[i+1,j],   length(ps)))
        push!(faces, (Li[i+1,j],   Li[i+1,j+1], length(ps)))
        push!(faces, (Li[i+1,j+1], Li[i,j+1],   length(ps)))
        push!(faces, (Li[i,j+1],   Li[i,j],     length(ps)))
    end

    verticies = permutedims(reinterpret(reshape,eltype(eltype(ps)), ps))
    faces = permutedims(reinterpret(reshape,Int, faces))

    return verticies, faces, values
end


function make_plot(g,gf)
    v,f,c = verticies_and_faces_and_values(g,gf)
    mesh(v,f,color=c,
        shading = NoShading,
    )
end

# scatter(collect(g)[:])

function Makie.surface(g::Grid{<:Any,2}, gf::AbstractArray{<:Any, 2}; kwargs...)
    surface(getindex.(g,1), getindex.(g,2), gf;
        shading = NoShading,
        kwargs...,
    )
end

function Makie.mesh(g::Grid{<:Any,2}, gf::AbstractArray{<:Any, 2}; kwargs...)
    v,f,c = verticies_and_faces_and_values(g, gf)
    mesh(v,f,color=c,
        shading = NoShading,
        kwargs...,
    )
end

function Makie.plot!(plot::Plot(Grid{<:Any,2},AbstractArray{<:Any, 2}))
    # TODO: How to handle kwargs?
    # v,f,c = verticies_and_faces_and_values(plot[1], plot[2])
    r = @lift verticies_and_faces_and_values($(plot[1]), $(plot[2]))
    v,f,c = (@lift $r[1]), (@lift $r[2]), (@lift $r[3])
    mesh!(plot, v, f, color=c,
        shading = NoShading,
    )
end

Makie.convert_arguments(::Type{<:Scatter}, g::Grid) = (map(Tuple,collect(g)[:]),)
end
