module SbplibMakieExt

using Sbplib.Grids
using Makie
using StaticArrays


function verticies_and_faces_and_values(g::Grid{<:Any,2}, gf::AbstractArray{<:Any, 2})
    ps = map(Tuple, g)[:]
    values = gf[:]
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


## Grids

Makie.convert_arguments(::Type{<:Scatter}, g::Grid) = (reshape(map(Point,g),:),) # (map(Point,collect(g)[:]),)
function Makie.convert_arguments(::Type{<:Lines}, g::Grid{<:Any,2})
    M = collect(g)

    function cat_with_NaN(a,b)
        vcat(a,[@SVector[NaN,NaN]],b)
    end

    xlines = reduce(cat_with_NaN, eachrow(M))
    ylines = reduce(cat_with_NaN, eachcol(M))

    return (cat_with_NaN(xlines,ylines),)
end

Makie.plot!(plot::Plot(Grid{<:Any,2})) = lines!(plot, plot.attributes, plot[1])


## Grid functions

### 1D
function Makie.convert_arguments(::Type{<:Lines}, g::Grid{<:Any,1}, gf::AbstractArray{<:Any, 1})
    (collect(g), gf)
end

function Makie.convert_arguments(::Type{<:Scatter}, g::Grid{<:Any,1}, gf::AbstractArray{<:Any, 1})
    (collect(g), gf)
end

Makie.plot!(plot::Plot(Grid{<:Any,1}, AbstractArray{<:Any,1})) = lines!(plot, plot.attributes, plot[1], plot[2])

### 2D
function Makie.convert_arguments(::Type{<:Surface}, g::Grid{<:Any,2}, gf::AbstractArray{<:Any, 2})
    (getindex.(g,1), getindex.(g,2), gf)
end

function Makie.plot!(plot::Plot(Grid{<:Any,2},AbstractArray{<:Any, 2}))
    r = @lift verticies_and_faces_and_values($(plot[1]), $(plot[2]))
    v,f,c = (@lift $r[1]), (@lift $r[2]), (@lift $r[3])
    mesh!(plot, plot.attributes, v, f;
        color=c,
        shading = NoShading,
    )
end
# TBD: Can we define `mesh` instead of the above function and then forward plot! to that?

function Makie.convert_arguments(::Type{<:Scatter}, g::Grid{<:Any,2}, gf::AbstractArray{<:Any, 2})
    ps = map(g,gf) do (x,y), z
        @SVector[x,y,z]
    end
    (reshape(ps,:),)
end

end
