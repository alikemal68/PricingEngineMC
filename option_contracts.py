
from abc import ABC, abstractmethod
import numpy as np


class Option(ABC):

    def __init__(self, strike, maturity):
        self.strike = strike
        self.maturity = maturity

    @abstractmethod
    def payoff(self, paths, times):
        pass


class EuropeanCall(Option):

    def payoff(self, paths, times):
        S_T = paths[:, -1]
        return np.maximum(S_T - self.strike, 0.0)


class EuropeanPut(Option):

    def payoff(self, paths, times):
        S_T = paths[:, -1]
        return np.maximum(self.strike - S_T, 0.0)
