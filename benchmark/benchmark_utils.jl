import PkgBenchmark
import Markdown
import Mustache
import Dates

import Sbplib

const sbplib_root = splitpath(pathof(Sbplib))[1:end-2] |> joinpath
const results_dir = mkpath(joinpath(sbplib_root, "benchmark/results"))
const template_path = joinpath(sbplib_root, "benchmark/result.tmpl")

"""
 main(args...; kwargs...)

Calls `run_benchmark(args...; kwargs...)` and writes the results as an HTML file in `benchmark/results`.
See [`run_benchmark`](@ref) for possible arguments.
"""
function main(args...; kwargs...)
    r = run_benchmark(args...; kwargs...)
    file_path = write_result_html(r)
    open_in_default_browser(file_path)
end


"""
    run_benchmark()

Runs the benchmark suite for the current working directory and returns a `PkgBenchmark.BenchmarkResult`
"""
function run_benchmark(;kwargs...)
    r = PkgBenchmark.benchmarkpkg(Sbplib; kwargs...)

    rev = hg_id()

    return add_rev_info(r, rev)
end

"""
    run_benchmark(rev)

Updates the repository to the given revison and runs the benchmark suite. When done, updates the repository to the origianl state.
`rev` can be any identifier compatible with `hg update`.

Returns a `PkgBenchmark.BenchmarkResult`
"""
function run_benchmark(rev; kwargs...)
    rev_before = hg_rev()
    hg_update(rev)
    r = run_benchmark(;kwargs...)
    hg_update(rev_before)

    return r
end

"""
    run_benchmark(target, baseline, f=minimum; judgekwargs=Dict())

Runs the benchmark at revisions `target` and `baseline` and compares them using `PkgBenchmark.judge`.
`f` is the function used to compare. `judgekwargs` are keyword arguments passed to `judge`.

`target` and `baseline` can be any identifier compatible with `hg update`.

Returns a `PkgBenchmark.BenchmarkJudgement`
"""
function run_benchmark(target, baseline, f=minimum; judgekwargs=Dict(), kwargs...)
    rev_before = hg_rev()
    hg_update(target)
    t = run_benchmark(;kwargs...)
    hg_update(baseline)
    b = run_benchmark(;kwargs...)
    hg_update(rev_before)

    return PkgBenchmark.judge(t,b,f; judgekwargs...)
end


function add_rev_info(benchmarkresult, rev)
    return PkgBenchmark.BenchmarkResults(
        benchmarkresult.name,
        rev,
        benchmarkresult.benchmarkgroup,
        benchmarkresult.date,
        benchmarkresult.julia_commit,
        benchmarkresult.vinfo,
        benchmarkresult.benchmarkconfig,
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
    dt = Dates.format(PkgBenchmark.date(r), "yyyy-mm-dd HHMMSS")
    file_path = joinpath(results_dir, dt*".html")

    open(file_path, "w") do io
        write_result_html(io, r)
    end

    return file_path
end


PkgBenchmark.date(j::PkgBenchmark.BenchmarkJudgement) = PkgBenchmark.date(PkgBenchmark.target_result(j))


function hg_id()
    cmd = Cmd(`hg id`, dir=sbplib_root)
    return readchomp(addenv(cmd, "HGPLAIN"=>""))
end

function hg_rev()
    cmd = Cmd(`hg id -i`, dir=sbplib_root)
    return readchomp(addenv(cmd, "HGPLAIN"=>""))
end

function hg_update(rev)
    cmd = Cmd(`hg update --check -r $rev`, dir=sbplib_root)
    run(addenv(cmd, "HGPLAIN"=>""))
end

function hg_commit(msg)
    cmd = Cmd(`hg commit --verbose --message $msg`, dir=sbplib_root)
    out = readchomp(addenv(cmd, "HGPLAIN"=>""))

    return only(match(r"committed changeset \d+:([0-9a-z]+)", out))
end

function hg_is_dirty()
    cmd = Cmd(`hg identify --id`, dir=sbplib_root)
    out = readchomp(addenv(cmd, "HGPLAIN"=>""))

    return endswith(out, "+")
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

# TODO: Better logging of what is happening
# TODO: Improve the workflow? How?

# TODO: Clean up the HTML output?
    # TODO: Make the codeblocks in the table look nicer
    # TODO: Change width of tables and code blocks so everything is visible
    # TODO: Fix the commit id, it chops off all the important info
    # TODO: Make title less verbose
    # TBD: Do we have to replace export_markdown? Could use a template instead.


# TBD: How to compare against current working directory? Possible to create a temporary commit?
    # Make a secret temporary commit
    # run benchmarkresult
    # uncommit the temporary commit
    # verbose flag on commit will print the revision id of the new commit.
# TBD: What parts are PkgBenchmark contributing? Can it be stripped out?
