abstract type TimeStepper end

# Returns v and t
function getState(ts::TimeStepper)
	error("not implemented")
end


function step!(ts::TimeStepper)
	error("not implemented")
end

function stepN(ts::TimeStepper,N::Int)
	for i ∈ 1:N
		ts.step()
	end
end

function stepTo(ts::TimeStepper)
	error("Not yet implemented")
end

function evolve(ts::TimeStepper)
	error("Not yet implemented")
end


mutable struct Rk4 <: TimeStepper
	F::Function
	k::Real
	v::Vector
	t::Real
	n::UInt

	function Rk4(F::Function,k::Real,v0::Vector,t0::Real)
		# TODO: Check that F has two inputs and one output
		v = v0
		t = t0
		n = 0
		return new(F,k,v,t,n)
	end
end

function getState(ts::Rk4)
	return ts.t, ts.v
end

function step!(ts::Rk4)
    k1 = ts.F(ts.v,ts.t)
	k2 = ts.F(ts.v+0.5*ts.k*k1,ts.t+0.5*ts.k)
	k3 = ts.F(ts.v+0.5*ts.k*k2,ts.t+0.5*ts.k)
    k4 = ts.F(ts.v+    ts.k*k3,ts.t+    ts.k)
    ts.v  = ts.v + (1/6)*(k1+2*(k2+k3)+k4)*ts.k

	ts.n = ts.n + 1
	ts.t = ts.t + ts.k

	return nothing
end

