
from abc import ABC, abstractmethod
import numpy as np


class StochasticModel(ABC):

    @abstractmethod
    def drift(self, t, s):
        pass

    @abstractmethod
    def diffusion(self, t, s):
        pass

    def diffusion_derivative(self, t, s):
        raise NotImplementedError(
            "Diffusion derivative not available"
        )

    def exact_step(self, t, s, dt, dW):
        raise NotImplementedError(
            "Exact transition not available"
        )
    

class GBM(StochasticModel):

    def __init__(self, r, sigma):
        self.r = r
        self.sigma = sigma

    def drift(self, t, s):
        return (self.r) * s

    def diffusion(self, t, s):
        return self.sigma * s

    def diffusion_derivative(self, t, s):
        # return np.full_like(s, self.sigma, dtype=float)
        return self.sigma
    
    def exact_step(self, t, s, dt, dW):
        return s * np.exp(
            (self.r  - 0.5*self.sigma**2)*dt
            + self.sigma*dW
        )

