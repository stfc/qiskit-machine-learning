"""Test the ``CovariantFeatureMap`` circuit."""

from test import QiskitMachineLearningTestCase
from qiskit_machine_learning.circuit.library import CovariantFeatureMap

class TestCovariantFeatureMap(QiskitMachineLearningTestCase):

    """Test the ``CovariantFeatureMap`` circuit."""

    def test_construction_with_training_parameters(self):
        """Test construction of ``CovariantFeatureMap``."""

        circuit = CovariantFeatureMap(
            feature_dimension=6,
            entanglement=[[0, 2], [2, 1]],
            include_training_parameters=True,
        )

        with self.subTest("check circuit built"):
            self.assertEqual(circuit.num_qubits, 3)
            self.assertEqual(len(circuit.input_parameters), 6)
            self.assertEqual(len(circuit.training_parameters), 3)
            self.assertEqual(circuit.num_parameters, 9)

    def test_construction_without_training_parameters(self):
        """Test construction of ``CovariantFeatureMap``."""

        circuit = CovariantFeatureMap(
            feature_dimension=6,
            entanglement=[[0, 2], [2, 1]],
            include_training_parameters=False,
        )

        with self.subTest("check circuit built"):
            self.assertEqual(circuit.num_qubits, 3)
            self.assertEqual(len(circuit.input_parameters), 6)
            self.assertEqual(len(circuit.training_parameters), 0)
            self.assertEqual(circuit.num_parameters, 6)


    def test_construction_fails(self):
        """Test invalid construction."""

        with self.assertRaisesRegex(ValueError, "even number"):
            CovariantFeatureMap(feature_dimension=3)

        with self.assertRaises(TypeError):
            CovariantFeatureMap()

        with self.assertRaises(IndexError):
            CovariantFeatureMap(
                feature_dimension=4,
                entanglement=[[0, 2]],
            )