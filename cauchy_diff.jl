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

q(X) = X[1] + X[2]^2 + cos(X[3])

q2 = x -> q(x*kd(2,3) .+ X.*(ones(3)-kd(2,3)))

q3 = x -> q(x*kd(3,3) .+ X.*(ones(3)-kd(3,3)))

function partial_nabla(f, i::Int64, X::Float64,delta::Float64)

  ## f:= the function to be differentiated
  ## i:= partial differentiation with respect to this index
  ## X:= where the partial derivative is evaluated
  ## delta:= the sampling frequency

  N = len(X)

  kd(i,n) = [j==i for j in 1:n]

  f_x = x -> f(x*kd(i,N) .+ X.*(ones(N)-kd(i,N)))

  return nabla(f_x,X[i],delta)

end