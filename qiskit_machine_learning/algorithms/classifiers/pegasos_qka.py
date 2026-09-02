"""Pegasos Quantum Kernel Alignment algorithm."""
from typing import Optional, Tuple, Union, Iterator

import numpy as np
from ...optimizers.spsa import bernoulli_perturbation
from .pegasos_qsvc import PegasosQSVC


class PegasosQKA(PegasosQSVC):
    r"""
    Extends the Pegasos Quantum Support Vector Classifier algorithm to perform quantum 
    kernel alignment within the optimization loop. The algorithm has been developed in [1]. 
    This implementation is adapted to work with trainable quantum kernels.

    **Example**

    .. code-block:: python

        quantum_kernel = TrainableFidelityQuantumKernel()

        pegasos_qka = PegasosQKA(quantum_kernel=quantum_kernel)
        pegasos_qka.fit(sample_train, label_train)
        pegasos_qka.predict(sample_test)

    **References**
        [1]: G. Gentinetta, D. Sutter, C. Zoufal, B. Fuller and S. Woerner, 
        Quantum Kernel Alignment with Stochastic Gradient Descent 
        <https://ieeexplore.ieee.org/document/10313634>`_

    """
     
    def __init__(
        self,
        quantum_kernel=None,
        C: float = 1000.0,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
        initial_thetas: Optional[np.ndarray] = None,
        learning_rate: float = 0.01,
        perturbations: float = 0.01,
    ) -> None:
        super().__init__(
            quantum_kernel=quantum_kernel,
            C=C,
            num_steps=num_steps,
            seed=seed,
        )

        if initial_thetas is None:
            self._theta = np.zeros(quantum_kernel.num_training_parameters)
        else:
            if len(initial_thetas) == quantum_kernel.num_training_parameters:
                self._theta = np.atleast_1d(initial_thetas)
            else:
                raise ValueError(
                    f"Number of parameters in initial guess ({len(initial_thetas)}) does not match"
                    f"number of parameters in Kernel ({quantum_kernel.num_training_parameters})."
                )

        self._support_thetas = {}
        self._left_theta = self._theta.copy()
        self.learning_rate = learning_rate
        self.perturbations = perturbations

    def _update_step(self, index: int, X: np.ndarray, y: np.ndarray, step: int) -> None:
        """
        Implements an update step for the fit method.
        
        Args:
            index: Index of the selected training sample.
            X: Training features.
            y: Training labels.
            step: Current training step.
        """
        value = self._compute_weighted_kernel_sum(index, X, training=True)

        y_step = self._label_map[y[index]]

        new_theta = self._theta
        support_theta=new_theta.copy()

        if (y_step * self.C / step) * value < 1:
  
            # choose update direction
            n = bernoulli_perturbation(self._quantum_kernel.num_training_parameters)

            # approximate gradient in that direction
            factor = y_step * self.C / step
            theta_plus = new_theta + self.perturbations * n
            theta_minus = new_theta - self.perturbations * n
            self._left_theta = theta_plus
            obj_plus = -factor * self._compute_weighted_kernel_sum(index, X, training=True)
            self._left_theta = theta_minus
            obj_minus = -factor * self._compute_weighted_kernel_sum(index, X, training=True)

            gradient = (obj_plus - obj_minus) / (2 * self.perturbations) if self.perturbations > 0 else 0

            new_theta = (new_theta - self.learning_rate * gradient * n).flatten()

            # update alpha & theta
            self._alphas[index] = self._alphas.get(index, 0) + 1
            if index not in self._support_thetas:
                self._support_thetas[index] = []

            self._support_thetas[index].append(support_theta)
            
        self._theta = new_theta
        self._left_theta = new_theta
    
    def _evaluate_kernel(self, x: np.ndarray, x_supp: np.ndarray, support_indices: list[int]) -> np.ndarray:
        """
        Evaluate the paramaterized kernel function for a single data point and the support vectors.

        For each support vector, evaluate the kernel using the current parameters
        on the left and for each parameter snapshot associated with that support vector on the right. 
        The resulting kernel values are averaged over the parameter snapshots.

        Args:
            x: Data point to evaluate.
            x_supp: Support vectors.
            support_indices: Training data indices corresponding to the support vectors x_supp.

        Returns:
            Kernel values between the data point and support vectors.
        """

        values = []
        for x_right, index in zip(x_supp, support_indices):
            kernels = []
            for right_theta in self._support_thetas[index]:
                kernel = self._quantum_kernel.evaluate(x, x_right, self._left_theta, right_theta) + self._kernel_offset
                kernels.append(kernel)
            values.append(np.mean(kernels))

        return np.asarray(values)

