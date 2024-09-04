using Documenter
using Sbplib

using Sbplib.DiffOps
using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.RegionIndices
using Sbplib.SbpOperators
using Sbplib.StaticDicts

sitename = "Sbplib.jl"

remotes = nothing
edit_link = nothing
repolink = nothing

if "--prettyurls" ∈ ARGS
    prettyurls = true
else
    prettyurls = false
end

if "--build-dir" ∈ ARGS
    i = findlast(==("--build-dir"), ARGS)
    build = ARGS[i+1]
else
    build = "build-local"
end

pages = [
    "Home" => "index.md",
    "operator_file_format.md",
    "grids_and_grid_functions.md",
    "matrix_and_tensor_representations.md",
    "Submodules" => [
        "submodules/grids.md",
        "submodules/diff_ops.md",
        "submodules/lazy_tensors.md",
        "submodules/region_indices.md",
        "submodules/sbp_operators.md",
        "submodules/static_dicts.md",
    ],
    "doc_index.md",
]
# This ordering is not respected by @contents. See https://github.com/JuliaDocs/Documenter.jl/issues/936

format=Documenter.HTML(;prettyurls, edit_link, repolink)

makedocs(;sitename, pages, format, build, remotes)
