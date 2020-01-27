function nabla(f, x::Float64, delta::Float64)

  ## differentiation of holomorphic functions in a single complex variable applied 
  ## to real-valued functions in a single variable using the Cauchy Integral Formula

  ## f:= the function to be differentiated
  ## x:= where the derivative is evaluated
  ## delta:= the sampling frequency

  N = round(Int,2*pi/delta)
  thetas = vcat(1:N)*delta

  ## collect arguments and rotations: 
  rotations = map(theta -> exp(-im*theta),thetas)
  arguments = x .+ conj.(rotations)  

  ## calculate expectation: 
  expectation = 1.0/N*real(sum(map(f,arguments).*rotations))

  return expectation

end

function partial_nabla(f, i::Int64, X::Array{Float64,1},delta::Float64)

  ## partial differentiation of holomorphic functions in a single complex variable applied 
  ## to real-valued functions in a single variable using the Cauchy Integral Formula

  ## f:= the function to be differentiated
  ## i:= partial differentiation with respect to this index
  ## X:= where the partial derivative is evaluated
  ## delta:= the sampling frequency

  N = length(X)

  kd(i,n) = [j==i for j in 1:n]

  f_i = x -> f(x*kd(i,N) .+ X.*(ones(N)-kd(i,N)))

  return nabla(f_i,X[i],delta)

end

function mc_nabla(f, x::Float64, delta::Float64)

  ## differentiation of holomorphic functions in a single complex variable applied 
  ## to real-valued functions in a single variable using monte carlo Contour Integration

  ## f:= the function to be differentiated
  ## x:= where the derivative is evaluated
  ## delta:= the sampling frequency

  N = round(Int,2*pi/delta)

  ## sample with only half the number of points: 
  sample = rand(1:N,round(Int,N/2)) 
  thetas = sample*delta

  ## collect arguments and rotations: 
  rotations = map(theta -> exp(-im*theta),thetas)
  arguments = x .+ conj.(rotations)  

  ## calculate expectation: 
  expectation = (2.0/N)*real(sum(map(f,arguments).*rotations))

  return expectation

end

function jacobian(f,X::Array{Float64,1},delta::Float64)
    
    N = Int(length(X))
    
    ## initialise jacobian: 
    J = zeros(N,N)
    
    for i = 1:N
        
        f_i(x) = f(x)[i]
        J[i,:] = [partial_nabla(f_i,j,X,delta) for j=1:N]
        
    end
    
    return J
    
end