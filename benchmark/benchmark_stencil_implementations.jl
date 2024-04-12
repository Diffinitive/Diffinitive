include("benchmark_utils.jl")

baseline       = run_benchmark("baseline", retune=true)
mapreduce      = run_benchmark("mapreduce")
multiplication = run_benchmark("multiplication")
promote_op     = run_benchmark("promote_op")
return_type    = run_benchmark("return_type")
sum            = run_benchmark("sum")
# TODO: Change to rev-ids

f = minimum

judge_mapreduce      = PkgBenchmark.judge(mapreduce,      baseline, f)
judge_multiplication = PkgBenchmark.judge(multiplication, baseline, f)
judge_promote_op     = PkgBenchmark.judge(promote_op,     baseline, f)
judge_return_type    = PkgBenchmark.judge(return_type,    baseline, f)
judge_sum            = PkgBenchmark.judge(sum,            baseline, f)

write_result_html(judge_mapreduce,      name="mapreduce")
write_result_html(judge_multiplication, name="multiplication")
write_result_html(judge_promote_op,     name="promote_op")
write_result_html(judge_return_type,    name="return_type")
write_result_html(judge_sum,            name="sum")
