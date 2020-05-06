def t_grad(fun,grad_fun,x,dx,ind):
    try:
        N = len(x)
        xL = True
    except:
        xL = False
        
    if xL:
        y1 = fun(x)
        x[ind] += dx
        y2 = fun(x)

        x[ind] -= 0.5*dx
        g = grad_fun(x)
        return (y2-y1)/dx,g[ind]
    else:
        y1 = fun(x)
        x += dx
        y2 = fun(x)

        x -= 0.5*dx
        g = grad_fun(x)
        return (y2-y1)/dx,g
        
    

