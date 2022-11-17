module Lipids2D
export lipidbilayer
using Ipopt
using JuMP
using Plots

function kernel(s)
    -0.5*exp.(-abs.(s))
end

function lipidbilayer(L, dx, lipidlength, c0, m; 
    cmin=1e-5, alpha=1, sigma = 5, gamma=0.7)

    # Set up problem domain
    N = Int(2*L/dx + 1)
    x = -L:dx:L
    y = -L:dx:L
    d = m/dx^2

    p = Int(lipidlength/dx)
    px = [p, 0, -p, 0]
    py = [0, p, 0, -p]

    model = Model(Ipopt.Optimizer)

    # function kappa(i, j, m, n)
    #     s = hypot(
    #         min(mod(i - m, N), mod(m - i, N)),
    #         min(mod(j - n, N), mod(n - j, N))
    #     )
    #     return kernel(s)
    # end

    #function tails(u, i, j)
    #    return sum(u[i, j, k] for k = 1:4)
    #end

    #function heads(u, i, j)
    #    return sum(
    #        u[mod(i - px[k] - 1, N) + 1, mod(j - py[k] - 1, N) + 1, k] 
    #        for k = 1:4
    #    )
    #end

   #  function water(u, i, j)
   #      return 1 - heads(u, i, j) - tails(u, i, j)
   #  end
    
    @variable(model, u[1:N, 1:N, 1:4] >= cmin)

    # Set up macros for heads, tails, and water
    @expression(model, tails[i=1:N, j=1:N], sum(u[i, j, k] for k=1:4))
    @expression(model, heads[i=1:N, j=1:N], 
        sum(
            u[mod(i - px[k] - 1, N) + 1, mod(j - py[k] - 1, N) + 1, k] 
            for k = 1:4
        )
    )
    @expression(model, water[i=1:N, j=1:N], heads[i,j] + tails[i, j])

    @constraint(model, sum(u) == d)
    @constraint(model,
        [i = 1:N, j = 1:N],
        heads[i,j] + tails[i, j] ≤ 1
    )
    @NLexpression(model, kappa[i=1:N, j=1:N, m=1:N, n=1:N], 
        -0.5*exp(-(
            min(mod(i - m, N), mod(m - i, N))^2,
            min(mod(j - n, N), mod(n - j, N))^2
        )^0.5))
    @NLobjective(
        model,
        Min,
        sum(
            tails[i, j] * log(tails[i, j]) +
            alpha * water[i, j] * sum(
                kappa[i, j, m, n] * (water[m, n] + gamma*heads[m, n])
                for m = 1:N, n = 1:N if kappa[i, j, m, n] < -1e-3
            )
            for i = 1:N, j = 1:N
        )*dx
    )
    print(model)
end
end



