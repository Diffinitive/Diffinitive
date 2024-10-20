import PkgBenchmark
import Markdown
import Mustache
import Dates

import Diffinitive

const diffinitive_root = splitpath(pathof(Diffinitive))[1:end-2] |> joinpath
const results_dir = mkpath(joinpath(diffinitive_root, "benchmark/results"))
const template_path = joinpath(diffinitive_root, "benchmark/result.tmpl")

"""
    mainmain(;rev=nothing, target=nothing, baseline=nothing , kwargs...)

Calls `run_benchmark(args...; kwargs...)` and writes the results as an HTML
file in `benchmark/results`.

 * If `rev` is set, the benchmarks are run for the given mercurial revision.
 * If only `baseline` is set, the current working directory is compared with
   the revision given in `baseline`.
 * If  both `target` and `baseline` is set those revision are compared.

For control over what happens to the benchmark result datastructure see the
different methods of [`run_benchmark`](@ref)
"""
function main(;rev=nothing, target=nothing, baseline=nothing, name=nothing, kwargs...)
    if !isnothing(rev)
        r = run_benchmark(rev; kwargs...)
    elseif !isnothing(baseline)
        if isnothing(target)
            r = compare_benchmarks(baseline; kwargs...)
        else
            r = compare_benchmarks(target, baseline; kwargs...)
        end
    else
        # Neither rev, or baseline were set => Run on current working directory.
        r = run_benchmark(;kwargs...)
    end

    file_path = write_result_html(r; name)
    open_in_default_browser(file_path)
end


"""
    run_benchmark()

Run the benchmark suite for the current working directory and return a
`PkgBenchmark.BenchmarkResult`
"""
function run_benchmark(;kwargs...)
    r = PkgBenchmark.benchmarkpkg(Diffinitive; kwargs...)

    rev = hg_rev() # Should be changed to hg_id() when the html can handle it.

    return add_rev_info(r, rev)
end

"""
    run_benchmark(rev)

Updates the repository to the given revison and runs the benchmark suite. When
done, reverts the repository to the original state. `rev` can be any
identifier compatible with `hg update`.

Returns a `PkgBenchmark.BenchmarkResult`
"""
function run_benchmark(rev; kwargs...)
    return hg_at_revision(rev) do
        run_benchmark(; kwargs...)
    end
end

"""
    compare_benchmarks(target, baseline, f=minimum; judgekwargs=Dict())

Runs the benchmark at revisions `target` and `baseline` and compares them
using `PkgBenchmark.judge`. `f` is the function used to compare. `judgekwargs`
are keyword arguments passed to `judge`.

`target` and `baseline` can be any identifier compatible with `hg update`.

Returns a `PkgBenchmark.BenchmarkJudgement`
"""
function compare_benchmarks(target, baseline, f=minimum; judgekwargs=Dict(), kwargs...)
    t = run_benchmark(target; kwargs...)
    b = run_benchmark(baseline; kwargs...)

    return PkgBenchmark.judge(t,b,f; judgekwargs...)
end

"""
    compare_benchmarks(baseline, ...)

Compare the results at the current working directory with the revision
specified in `baseline`.

Accepts the same arguments as the two revision version.
"""
function compare_benchmark(baseline, f=minimum; judgekwargs=Dict(), kwargs...)
    t = run_benchmark(;kwargs...)
    b = run_benchmark(baseline; kwargs...)

    return PkgBenchmark.judge(t,b,f; judgekwargs...)
end


function add_rev_info(benchmarkresult, rev)
    if endswith(rev,"+")
        revstr = "+$rev" # Workaround for the bad presentation of BenchmarkResults.
    else
        revstr = rev
    end

    return PkgBenchmark.BenchmarkResults(
        benchmarkresult.name,
        revstr,
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

function write_result_html(r; name=nothing)
    dt = Dates.format(PkgBenchmark.date(r), "yyyy-mm-dd HHMMSS")

    if isnothing(name)
        file_path = joinpath(results_dir, dt*".html")
    else
        file_path = joinpath(results_dir, dt*" "*name*".html")
    end

    open(file_path, "w") do io
        write_result_html(io, r)
    end

    return file_path
end


PkgBenchmark.date(j::PkgBenchmark.BenchmarkJudgement) = PkgBenchmark.date(PkgBenchmark.target_result(j))


function hg_id()
    cmd = Cmd(`hg id`, dir=diffinitive_root)
    return readchomp(addenv(cmd, "HGPLAIN"=>""))
end

function hg_rev()
    cmd = Cmd(`hg id -i`, dir=diffinitive_root)
    return readchomp(addenv(cmd, "HGPLAIN"=>""))
end

function hg_update(rev)
    cmd = Cmd(`hg update --check -r $rev`, dir=diffinitive_root)
    run(addenv(cmd, "HGPLAIN"=>""))

    return nothing
end

"""
    hg_commit(msg; secret=false)

Make a hg commit with the provided message. If `secret` is true the commit is
in the secret phase stopping it from being pushed.
"""
function hg_commit(msg; secret=false)
    if secret
        cmd = Cmd(`hg commit --verbose --secret --message $msg`, dir=diffinitive_root)
    else
        cmd = Cmd(`hg commit --verbose          --message $msg`, dir=diffinitive_root)
    end

    out = readchomp(addenv(cmd, "HGPLAIN"=>""))

    return only(match(r"committed changeset \d+:([0-9a-z]+)", out))
end

"""
    hg_strip(rev; keep=false)

Strips the given commit from the repo. If `keep` is true, the changes of the
commit are kept in the working directory.
"""
function hg_strip(rev; keep=false)
    if keep
        cmd = Cmd(`hg --config extensions.strip= strip --keep -r $rev`, dir=diffinitive_root)
    else
        cmd = Cmd(`hg --config extensions.strip= strip        -r $rev`, dir=diffinitive_root)
    end

    run(addenv(cmd, "HGPLAIN"=>""))

    return nothing
end

"""
    hg_is_dirty()

Return true if the repositopry has uncommited changes.
"""
function hg_is_dirty()
    cmd = Cmd(`hg identify --id`, dir=diffinitive_root)
    out = readchomp(addenv(cmd, "HGPLAIN"=>""))

    return endswith(out, "+")
end

"""
    hg_at_revision(f, rev)

Update the repository to the given revision and run the function `f`. After
`f` is run the working directory is restored. If there are uncommited changes
a temporary commit will be used to save the state of the working directory.
"""
function hg_at_revision(f, rev)
    if hg_is_dirty()
        hg_with_temporary_commit() do
            return _hg_at_revision(f, rev)
        end
    else
        return _hg_at_revision(f, rev)
    end
end

function _hg_at_revision(f, rev)
    @assert !hg_is_dirty()

    origin_rev = hg_rev()

    hg_update(rev)
    try
        return f()
    finally
        hg_update(origin_rev)
    end
end

"""
    hg_with_temporary_commit(f)

Run the function `f` after making a temporary commit with the current working
directory. After `f` has finished the working directory is restored to its
original state and the temporary commit stripped.
"""
function hg_with_temporary_commit(f)
    @assert hg_is_dirty()

    origin_rev = hg_commit("[Automatic commit by julia]",secret=true)

    try
        return f()
    finally
        hg_update(origin_rev)
        hg_strip(origin_rev; keep=true)
    end
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

# Should be able to run the current benchmark script at a different revision.
# Should have a way to filter the benchmark suite

# TBD: What parts are PkgBenchmark contributing? Can it be stripped out?


## Catching the exit code and errors from a command can be done with code similar to
    # proc = open(cmd)
    # if success(proc)

    # else

    # end
