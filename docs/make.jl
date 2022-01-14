using Documenter
using Sbplib

using Sbplib.DiffOps
using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.RegionIndices
using Sbplib.SbpOperators
using Sbplib.StaticDicts

sitename = "Sbplib.jl"

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
    "index.md",
    "operator_file_format.md",
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

format=Documenter.HTML(;prettyurls)
makedocs(;sitename, pages, format, build)
