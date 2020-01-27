f(W,x) = [sum(W[1,:].*x) ,sum(W[2,:].*x),sum(W[3,:].*x)] .+W[4,:]

g(Z) = [sum(Z.*X) ,sum(W[2,:].*X),sum(W[3,:].*X)] .+W[4,:]

function gradient_updates(W::Array{Float64,2},X::Array{Float64,1},case::Int64,delta::Float64)

	relu(x) = log.(1 .+ exp.(x))

	if case == 1

		g(z) = relu([sum(W[1,:].*z) ,sum(W[2,:].*z),sum(W[3,:].*z)] .+W[4,:])

		return jacobian(g,X,delta)

	else 

		## there is probably a way to simplify this: 
		g_1(Z) = relu([sum(Z.*X) ,sum(W[2,:].*X),sum(W[3,:].*X)] .+W[4,:])

		g_2(Z) = relu([sum(W[1,:].*X) ,sum(Z.*X),sum(W[3,:].*X)] .+W[4,:])

		g_3(Z) = relu([sum(W[1,:].*X) ,sum(W[2,:].*X),sum(Z.*X)] .+W[4,:])

		g_4(Z) = relu([sum(W[1,:].*X) ,sum(W[2,:].*X),sum(W[3,:].*X)] .+Z)

		return [jacobian(g_1,W[1,:],delta), jacobian(g_2,W[2,:],delta),jacobian(g_3,W[3,:],delta),jacobian(g_4,W[4,:],delta)]

	end

end

function jacobian(f,X::Array{Float64,1},case::Int64,delta::Float64)
    
    N = Int(length(X))
    
    if case == 1
    
        ## initialise jacobian: 
        J = zeros(N,N)

        for i = 1:N

            f_i(x) = f(x)[i]
            J[i,:] = [partial_nabla(f_i,j,X,delta) for j=1:N]

        end

        return J
        
    else

    	J = [partial_nabla(f,j,X,delta) for j=1:N]
            
       return J

   end
        
end