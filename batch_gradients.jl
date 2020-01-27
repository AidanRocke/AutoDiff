using Distributions

function GD(x)
    
    ## delta: 
    delta = 2*pi/100
    
    ## We define a linear function: 
    #W = rand(3,2)
    
    ## we use Xavier initialisation: 
    W = rand(Uniform(-1/sqrt(3),1/sqrt(3)),(3,2))

    f(Z) = [sum(W[1,:].*Z) ,sum(W[2,:].*Z),sum(W[3,:].*Z)] .+W[4,:]

    ## our squared loss: 
    L(x) = sum((x .- f(x)).^2)

    alpha = 0.1

    for i = 1:50

    	X = 10*rand(2)
    
        ## gradient updates:         
        dL = alpha*jacobian(L,X,2,2*pi/100)'
    
        dW_ = gradient_updates(W,X,2,delta)
    
        W[1,:] -= (alpha*dL*dW_[1])'

		W[2,:] -= (alpha*dL*dW_[2])'

		W[3,:] -= (alpha*dL*dW_[3])'

		W[4,:] -= (alpha*dL*dW_[4])'

    end

    return W
    
end