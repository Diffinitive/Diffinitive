using PkgBenchmark
using Markdown
using Mustache

import Sbplib

# From Pluto.jl/src/webserver/WebServer.jl  (2023-01-24)
function open_in_default_browser(url::AbstractString)::Bool
    try
        if Sys.isapple()
            Base.run(`open $url`)
            true
        elseif Sys.iswindows() || detectwsl()
            Base.run(`powershell.exe Start "'$url'"`)
            true
        elseif Sys.islinux()
            Base.run(`xdg-open $url`)
            true
        else
            false
        end
    catch ex
        false
    end
end

r = benchmarkpkg(Sbplib)

iobuffer = IOBuffer()
export_markdown(iobuffer, r)

parsed_md = Markdown.parse(String(take!(iobuffer)))

sbplib_root = splitpath(pathof(Sbplib))[1:end-2] |> joinpath

results_dir = mkpath(joinpath(sbplib_root, "benchmark/results"))

dt = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

file_path = joinpath(results_dir, dt*".html")


template = Mustache.load(joinpath(sbplib_root, "benchmark/result.tmpl"))

open(file_path, "w") do io
    content = html(parsed_md)
    Mustache.render(io, template, Dict("title"=>dt, "content"=>content))
    # html(io, parsed_md)
end

open_in_default_browser(file_path)


# TODO: Cleanup code
# TODO: Change color of codeblocks
# TODO: Change width of tables and code blocks
