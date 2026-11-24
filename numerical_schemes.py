
def euler_step(model, t, s, dt, dW):
    a = model.drift(t, s)
    b = model.diffusion(t, s)
    
    return (
        s + a * dt + b * dW
    )


def milstein_step(model, t, s, dt, dW):
    a = model.drift(t, s)
    b = model.diffusion(t, s)
    b_s = model.diffusion_derivative(t, s)

    return (
        s + a * dt + b * dW
        + 0.5 * b * b_s * (dW**2 - dt)
    )
