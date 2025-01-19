from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


### Abstract classes


class Sketch(ABC):
    def __init__(self, shape: Tuple[int, int]) -> None:
        self.shape = shape

    @abstractmethod
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        pass


class SketchFactory(ABC):
    def __init__(self, shape: Tuple[int, int]) -> None:
        self.shape = shape

    @abstractmethod
    def __call__(self) -> Sketch:
        pass


### Implementations


class MatrixSketch(Sketch):
    def __init__(self, S: np.ndarray) -> None:
        super().__init__(S.shape)
        self.S = S
        self.T = S.T

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.S @ x


class GaussianSketchFactory(SketchFactory):
    def __init__(self, shape: Tuple[int, int], rng: np.random.Generator = None) -> None:
        super().__init__(shape)
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def __call__(self) -> Sketch:
        S = self.rng.normal(size=self.shape) / np.sqrt(self.shape[0])
        return MatrixSketch(S)


class TransposedSubsamplingSketch(Sketch):
    def __init__(self, S: np.ndarray, n : int) -> None:
        super().__init__((n,S.shape[0]))
        self.S = S

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        new_shape = (self.shape[0],x.shape[1]) if len(x.shape) > 1 else (self.shape[0])
        y = np.zeros(new_shape)
        y[self.S,...] = x
        return y
    
class SubsamplingSketch(Sketch):
    def __init__(self, S: np.ndarray, n : int) -> None:
        super().__init__((S.shape[0],n))
        self.S = S
        self.T = TransposedSubsamplingSketch(self.S, self.shape[1])

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return x[self.S,...]


    
### Subsampling according to given probabilites. Default to uniform sampling.

class SubsamplingSketchFactory(SketchFactory):
    def __init__(
        self,
        shape: Tuple[int, int],
        probabilities: np.ndarray = None,
        rng: np.random.Generator = None
    ) -> None:
        super().__init__(shape)
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.probabilities = probabilities

        if self.probabilities is not None:
            total_prob = np.sum(self.probabilities)
            self.probabilities = self.probabilities / total_prob   # Normalize probabilities to sum up to 1.


    def __call__(self) -> Sketch:
        S = self.rng.choice(np.arange(self.shape[1]), self.shape[0], replace=False, p=self.probabilities)
        return SubsamplingSketch(S,self.shape[1])
