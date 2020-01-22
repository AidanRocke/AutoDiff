# AutoDiff
Automatic Differentiation via Contour Integration

## Motivation: 

There has previously been some back-and-forth among scientists about whether biological networks such as brains
might compute derivatives. I have previously made my position on this issue clear: https://twitter.com/bayesianbrain/status/1202650626653597698

The standard counter-argument is that backpropagation isn't biologically plausible
but partial derivatives are very useful for closed-loop control so we are faced with a fundamental question we
can't ignore. How might large branching structures in the brain and other biological systems compute derivatives?

After some reflection I realised that an important result in complex analysis due to Cauchy, the Cauchy Integral Formula, 
may be used to compute derivatives with a simple forward propagation of signals using a monte-carlo method. Incidentally, 
Cauchy also discovered the gradient descent algorithm. 

**Minimal implementation of differentiation via Contour Integration in the Julia language:**

```julia
function nabla(f, x::Float64, delta::Float64)

  ## automatic differentiation of holomorphic functions in a single complex variable
  ## applied to real-valued functions in a single variable using the Cauchy Integral Formula

  N = round(Int,2*pi/delta)
  thetas = vcat(1:N)*delta

  ## collect arguments and rotations: 
  rotations = map(theta -> exp(-im*theta),thetas)
  arguments = x .+ conj.(rotations)  

  ## calculate expectation: 
  expectation = 1.0/N*real(sum(map(f,arguments).*rotations))

  return expectation

end
```

**Minimal implementation of partial differentiation via Contour Integration in the Julia language:**

```julia
function partial_nabla(f, i::Int64, X::Array{Float64,1},delta::Float64)

  ## f:= the function to be differentiated
  ## i:= partial differentiation with respect to this index
  ## X:= where the partial derivative is evaluated
  ## delta:= the sampling frequency

  N = length(X)

  kd(i,n) = [j==i for j in 1:n]

  f_i = x -> f(x*kd(i,N) .+ X.*(ones(N)-kd(i,N)))

  return nabla(f_i,X[i],delta)

end
```

**Blog post:**

https://keplerlounge.com/neural-computation/2020/01/16/complex-auto-diff.html

**Jupyter Notebook:**

1. [Main tutorial](https://github.com/AidanRocke/AutoDiff/blob/master/main_tutorial.ipynb)

2. [Physics simulations](https://github.com/AidanRocke/AutoDiff/blob/master/physics_simulations.ipynb)

3. [Convergence of Error](https://github.com/AidanRocke/AutoDiff/blob/master/convergence_of_error.ipynb)

## Supplementary blog posts: 

1. [An alternative definition for the Partial Derivative](https://keplerlounge.com/applied-math/2020/01/20/partial-derivative.html)

