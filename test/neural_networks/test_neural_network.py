# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Neural Network."""

import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import NeuralNetwork


class _NeuralNetwork(NeuralNetwork):
    """Dummy implementation to test the abstract neural network class."""

    def __init__(
        self,
        num_inputs,
        num_weights,
        sparse,
        output_shape,
        input_gradients=False,
        pass_manager=None,
    ) -> None:
        self.pass_manager = pass_manager
        super().__init__(
            num_inputs=num_inputs,
            num_weights=num_weights,
            sparse=sparse,
            output_shape=output_shape,
            input_gradients=input_gradients,
            pass_manager=pass_manager,
        )

    def _forward(self, input_data, weights, input_params=None):
        """Expects as input either None, or a 2-dim array and returns."""

        # we add a dummy batch dimension
        batch_size = input_data.shape[0] if input_data is not None else 1
        return np.zeros((batch_size, *self.output_shape))

    def _backward(self, input_data, weights, input_params=None):
        # return None if there are no weights
        input_grad = None
        # we add a dummy batch down below
        batch_size = input_data.shape[0] if input_data is not None else 1
        if self.num_inputs > 0:
            input_grad = np.zeros((batch_size, *self.output_shape, self.num_inputs))

        weight_grad = None
        if self.num_weights > 0:
            weight_grad = np.zeros((batch_size, *self.output_shape, self.num_weights))

        return input_grad, weight_grad


@ddt
class TestNeuralNetwork(QiskitMachineLearningTestCase):
    """Neural Network Tests."""

    def __init__(
        self,
        TestCase,
    ):
        self.backend = GenericBackendV2(num_qubits=3, seed=123)
        self.pass_manager = generate_preset_pass_manager(backend=self.backend, optimization_level=0)
        super().__init__(TestCase)

    @staticmethod
    def _get_batch_size(input_data):
        batch_size = 1
        # if we have list of lists then the batch size is the length of the first list
        if isinstance(input_data, list) and isinstance(input_data[0], list):
            batch_size = len(input_data)
        return batch_size

    @data(
        # no input
        ((0, 0, True, 1), None),
        ((0, 1, True, 1), None),
        ((0, 1, True, 2), None),
        ((0, 1, True, (2, 2)), None),
        # 1d input
        ((1, 0, True, 1), 0),
        ((1, 1, True, 1), 0),
        ((1, 1, True, 2), 0),
        ((1, 1, True, (2, 2)), 0),
        # multi-dimensional input and weights
        ((2, 2, True, (2, 2)), [0, 0]),
        # batch test
        ((2, 2, True, (2, 2)), [[0, 0], [1, 1]]),
    )
    def test_forward_shape(self, params):
        """Test forward shape."""

        config, input_data = params
        batch_size = self._get_batch_size(input_data)
        network = _NeuralNetwork(*config)

        shape = network.forward(input_data, np.zeros(network.num_weights)).shape
        self.assertEqual(shape, (batch_size, *network.output_shape))

    @data(
        # no input
        ((0, 0, True, 1), None),
        ((0, 1, True, 1), None),
        ((0, 1, True, 2), None),
        ((0, 1, True, (2, 2)), None),
        # 1d input
        ((1, 0, True, 1), 0),
        ((1, 1, True, 1), 0),
        ((1, 1, True, 2), 0),
        ((1, 1, True, (2, 2)), 0),
        # multi-dimensional input and weights
        ((2, 2, True, (2, 2)), [0, 0]),
        # batch test
        ((2, 2, True, (2, 2)), [[0, 0], [1, 1]]),
    )
    def test_backward_shape(self, params):
        """Test backward shape"""

        config, input_data = params
        batch_size = self._get_batch_size(input_data)
        network = _NeuralNetwork(*config)

        input_grad, weights_grad = network.backward(input_data, np.zeros(network.num_weights))

        if network.num_inputs > 0:
            self.assertEqual(
                input_grad.shape,
                (batch_size, *network.output_shape, network.num_inputs),
            )
        else:
            self.assertEqual(input_grad, None)

        if network.num_weights > 0:
            self.assertEqual(
                weights_grad.shape,
                (batch_size, *network.output_shape, network.num_weights),
            )
        else:
            self.assertEqual(weights_grad, None)

    def test_data_gradients(self):
        """Tests data_gradient setter/getter."""
        network = _NeuralNetwork(1, 1, True, 1)
        self.assertFalse(network.input_gradients)

        network.input_gradients = True
        self.assertTrue(network.input_gradients)

        network = _NeuralNetwork(1, 1, True, 1, True)
        self.assertTrue(network.input_gradients)

    def test_compose_circs_no_layout(self):
        """Neither circuit has layout._input_qubit_count: simple sequential compose."""
        network = _NeuralNetwork(0, 0, True, 1)
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        # add an op to each so we can count
        qc1.x(0)
        qc2.z(0)
        comp = network._compose_circs(qc1, qc2)
        self.assertIsInstance(comp, QuantumCircuit)
        self.assertEqual(comp.num_qubits, 1)
        # x followed by z
        self.assertEqual([op.operation.name for op in comp.data], ["x", "z"])

    def test_compose_circs_matching_layout(self):
        """Both circuits have matching layout._input_qubit_count: direct compose."""
        network = _NeuralNetwork(0, 0, True, 1, pass_manager=self.pass_manager)
        qc1 = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        isa_qc1 = self.pass_manager.run(qc1)
        isa_qc2 = self.pass_manager.run(qc2)
        isa_qc1.layout._input_qubit_count = 2
        isa_qc2.layout._input_qubit_count = 2
        isa_qc1.h(0)
        isa_qc2.cx(0, 1)
        comp = network._compose_circs(isa_qc1, isa_qc2)
        self.assertEqual(comp.num_qubits, 3)
        names = [op.operation.name for op in comp.data]
        self.assertEqual(names, ["h", "cx"])

    def test_compose_circs_missing_pass_manager_raises(self):
        """Only ansatz has layout but no pass_manager: error."""
        network = _NeuralNetwork(0, 0, True, 1)
        qc_in = QuantumCircuit(1)
        qc_an = QuantumCircuit(1)
        qc_ans = self.pass_manager.run(qc_an)
        qc_ans.layout._input_qubit_count = 1
        if hasattr(network, "pass_manager"):
            del network.pass_manager
        with self.assertRaises(QiskitMachineLearningError):
            network._compose_circs(qc_in, qc_ans)

    def test_validate_input_none(self):
        """None input returns all Nones."""
        network = _NeuralNetwork(3, 3, True, 1)
        data, shape, params = network._validate_input(None, None)
        self.assertIsNone(data)
        self.assertIsNone(shape)
        self.assertIsNone(params)

    def test_validate_input_circuit_without_params(self):
        """Single QuantumCircuit without params."""
        network = _NeuralNetwork(0, 0, True, 1)
        qc = QuantumCircuit(2)
        data, shape, params = network._validate_input(qc, None)
        self.assertIs(data, qc)
        self.assertEqual(shape, (len(qc),))
        self.assertIsNone(params)

    def test_validate_input_circuit_with_params(self):
        """Single QuantumCircuit with a scalar param."""
        network = _NeuralNetwork(0, 0, True, 1)
        qc = QuantumCircuit(1)
        data, shape, params = network._validate_input(qc, 0.42)
        self.assertEqual(data, [qc])
        self.assertEqual(shape, (1, 1))
        self.assertTrue((params == np.array([[0.42]])).all())

    def test_validate_input_list_of_circuits(self):
        """List of QuantumCircuits without params."""
        network = _NeuralNetwork(0, 0, True, 1)
        qc1, qc2 = QuantumCircuit(1), QuantumCircuit(1)
        data, shape, params = network._validate_input([qc1, qc2], None)
        self.assertEqual(data, [qc1, qc2])
        self.assertEqual(shape, (2,))
        self.assertIsNone(params)

    def test_validate_input_scalar_and_arrays(self):
        """Numeric scalar, 1d and 2d inputs, and wrong-shape error."""
        net = _NeuralNetwork(1, 0, True, 1)
        d, s, p = net._validate_input(7.0, None)
        self.assertEqual(d.shape, (1, 1))
        self.assertEqual(s, ())
        # 1d correct
        net2 = _NeuralNetwork(3, 0, True, 1)
        d2, s2, p2 = net2._validate_input([1, 2, 3], None)
        self.assertEqual(d2.shape, (1, 3))
        self.assertEqual(s2, (3,))
        # 2d correct
        net3 = _NeuralNetwork(2, 0, True, 1)
        arr = np.array([[1, 2], [3, 4]])
        d3, s3, p3 = net3._validate_input(arr, None)
        self.assertEqual(d3.shape, (2, 2))
        self.assertEqual(s3, (2, 2))
        # mismatch
        with self.assertRaises(QiskitMachineLearningError):
            net3._validate_input([1, 2, 3], None)

    def test_preprocess_input_none(self):
        """Preprocess None input: repeats ansatz and empty params."""
        network = _NeuralNetwork(0, 0, True, 1)
        ansatz = QuantumCircuit(1)
        circs, vals, n, is_circ = network._preprocess_input(
            None, None, None, ansatz, output_shape=4
        )
        self.assertFalse(is_circ)
        self.assertEqual(n, 1)
        self.assertEqual(len(circs), 4)
        self.assertTrue(isinstance(vals, np.ndarray))
        self.assertEqual(vals.size, 0)

    def test_preprocess_input_numeric(self):
        """Preprocess numeric batch: repeats ansatz per sample."""
        network = _NeuralNetwork(2, 0, True, 1)
        ansatz = QuantumCircuit(2)
        batch = [[0.1, 0.2], [0.3, 0.4]]
        circs, vals, n, is_circ = network._preprocess_input(
            batch, None, None, ansatz, output_shape=3
        )
        self.assertFalse(is_circ)
        self.assertEqual(n, 2)
        self.assertEqual(len(circs), 6)
        self.assertTrue((vals == np.array(batch)).all())

    def test_preprocess_input_single_circuit(self):
        """Preprocess a single QuantumCircuit input."""
        network = _NeuralNetwork(0, 1, True, 1)
        ansatz = QuantumCircuit(1)
        qc = QuantumCircuit(1)
        circs, vals, n, is_circ = network._preprocess_input(qc, None, None, ansatz, output_shape=2)
        self.assertTrue(is_circ)
        self.assertEqual(n, 1)
        self.assertEqual(len(circs), 2)
        self.assertTrue(all(isinstance(c, QuantumCircuit) for c in circs))

    def test_preprocess_input_list_circuit(self):
        """Preprocess a list of QuantumCircuit inputs."""
        network = _NeuralNetwork(0, 1, True, 1)
        ansatz = QuantumCircuit(1)
        qc1, qc2 = QuantumCircuit(1), QuantumCircuit(1)
        circs, vals, n, is_circ = network._preprocess_input(
            [qc1, qc2], None, None, ansatz, output_shape=3
        )
        self.assertTrue(is_circ)
        self.assertEqual(n, 2)
        self.assertEqual(len(circs), 6)
        self.assertTrue(all(isinstance(c, QuantumCircuit) for c in circs))

    def test_preprocess_input_list_circuit_with_parameters(self):
        """Preprocess a list of parameterized QuantumCircuit inputs."""
        network = _NeuralNetwork(0, 2, True, 1)

        qc1, qc2 = QuantumCircuit(1), QuantumCircuit(1)
        params = [Parameter("p0"), Parameter("p1")]

        qc1.ry(params[0], 0)
        qc1.rx(params[1], 0)
        qc2.ry(params[0], 0)
        qc2.rx(params[1], 0)

        ansatz = QuantumCircuit(1)
        ansatz.ry(params[0], 0)
        ansatz.rx(params[1], 0)

        circuits, parameter_values, num_samples, is_circ_input = network._preprocess_input(
            [qc1, qc2], [-5, -6], [[1, 2], [3, 4]], ansatz, output_shape=3
        )

        self.assertTrue(is_circ_input)
        self.assertTrue(np.all(parameter_values == [[1, 2, -5, -6], [3, 4, -5, -6]]))
        self.assertEqual(num_samples, 2)
        self.assertEqual(len(circuits), 6)  # _circuits = [ansatz] * output_shape * num_samples
        self.assertTrue(all(isinstance(c, QuantumCircuit) for c in circuits))


if __name__ == "__main__":
    unittest.main()
