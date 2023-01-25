import PkgBenchmark
import Markdown
import Mustache
import Dates

import Sbplib

const sbplib_root = splitpath(pathof(Sbplib))[1:end-2] |> joinpath
const results_dir = mkpath(joinpath(sbplib_root, "benchmark/results"))
const template_path = joinpath(sbplib_root, "benchmark/result.tmpl")

function main()
    r = run_benchmark()
    file_path = write_result_html(r)
    open_in_default_browser(file_path)
end

function run_benchmark()
    r = PkgBenchmark.benchmarkpkg(Sbplib)

    commit = hg_id()

    return PkgBenchmark.BenchmarkResults(
        "Sbplib.jl",
        commit,
        r.benchmarkgroup,
        r.date,
        r.julia_commit,
        r.vinfo,
        r.benchmarkconfig,
    )
end

function write_result_html(io, r)
    iobuffer = IOBuffer()
    PkgBenchmark.export_markdown(iobuffer, r)

    parsed_md = Markdown.parse(String(take!(iobuffer)))
    content = Markdown.html(parsed_md)

    template = Mustache.load(template_path)

    dt = Dates.format(PkgBenchmark.date(r), "yyyy-mm-dd HH:MM:SS")
    Mustache.render(io, template, Dict("title"=>dt, "content"=>content))
end

function write_result_html(r)
    dt = Dates.format(PkgBenchmark.date(r), "yyyy-mm-dd HH:MM:SS")
    file_path = joinpath(results_dir, dt*".html")

    open(file_path, "w") do io
        write_result_html(io, r)
    end

    return file_path
end

function hg_id()
    cmd = Cmd(`hg id`, dir=sbplib_root)
    return readchomp(addenv(cmd, "HGPLAIN"=>""))
end

function hg_update(rev)
    cmd = Cmd(`hg update --check $rev`, dir=sbplib_root)
    run(addenv(cmd, "HGPLAIN"=>""))
end

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

main

# TODO: Change color of codeblocks
# TODO: Change width of tables and code blocks
