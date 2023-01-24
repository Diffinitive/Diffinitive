using BenchmarkTools
using Random

const SUITE = BenchmarkGroup()

SUITE["utf8"] = BenchmarkGroup(["string", "unicode"])
teststr = String(join(rand(MersenneTwister(1), 'a':'d', 10^4)))
SUITE["utf8"]["replace"] = @benchmarkable replace($teststr, "a" => "b")
SUITE["utf8"]["join"] = @benchmarkable join($teststr, $teststr)
SUITE["utf8"]["plots"] = BenchmarkGroup()

SUITE["trigonometry"] = BenchmarkGroup(["math", "triangles"])
SUITE["trigonometry"]["circular"] = BenchmarkGroup()
for f in (sin, cos, tan)
    for x in (0.0, pi)
        SUITE["trigonometry"]["circular"][string(f), x] = @benchmarkable ($f)($x)
    end
end

SUITE["trigonometry"]["hyperbolic"] = BenchmarkGroup()
for f in (sin, cos, tan)
    for x in (0.0, pi)
        SUITE["trigonometry"]["hyperbolic"][string(f), x] = @benchmarkable ($f)($x)
    end
end

SUITE


# TODO: Make it easy to run and display results (useful to look at them in a webbrowser? Could serve them using julia)
# TODO: Make it easy to compare different commits. (A simple script?)
# TODO: Do we need machanisms to save the results from runs?
# TBD: How well does BenchmarkTools work for comparisons, do we need make special considerations?
# TBD: When and how do we want to look at % of peak performance? Is this going to be done using the benchmark suite?


# Should set HGPLAIN before script use. Example: `HGPLAIN= hg st`
