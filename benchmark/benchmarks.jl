using BenchmarkTools
using Random

const SUITE = BenchmarkGroup()

SUITE["utf8"] = BenchmarkGroup(["string", "unicode"])
teststr = String(join(rand(MersenneTwister(1), 'a':'d', 10^4)))
SUITE["utf8"]["replace"] = @benchmarkable replace($teststr, "a" => "b")
SUITE["utf8"]["join"] = @benchmarkable join($teststr, $teststr)
SUITE["utf8"]["plots"] = BenchmarkGroup()

# SUITE["trigonometry"] = BenchmarkGroup(["math", "triangles"])
# SUITE["trigonometry"]["circular"] = BenchmarkGroup()
# for f in (sin, cos, tan)
#     for x in (0.0, pi)
#         SUITE["trigonometry"]["circular"][string(f), x] = @benchmarkable ($f)($x)
#     end
# end

# SUITE["trigonometry"]["hyperbolic"] = BenchmarkGroup()
# for f in (sin, cos, tan)
#     for x in (0.0, pi)
#         SUITE["trigonometry"]["hyperbolic"][string(f), x] = @benchmarkable ($f)($x)
#     end
# end

SUITE

# TODO: Add mercurial version of benchmarkpkg
# TODO: Make it easy to compare different commits. (A simple script?)

# Should set HGPLAIN before script use. Example: `HGPLAIN= hg st`
# `hg id` for getting the revision
# `hg update --check` for updating, requires a clean working directory.
