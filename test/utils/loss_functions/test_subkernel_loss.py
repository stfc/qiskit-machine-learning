# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from test import QiskitMachineLearningTestCase
from qiskit_machine_learning.utils.loss_functions.kernel_loss_functions import (
    BatchedSubKernelSVCLoss,
    SVCLoss,
)


class TestSubKernelLossFunctions(QiskitMachineLearningTestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(123)
        self.data = np.array(
            [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9],
            ]
        )

        self.labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    def test_sample_counts_preserve_ratio(self):
        loss = BatchedSubKernelSVCLoss(
            self.data, 
            self.labels, 
            sub_kernel_size=5
        )
        np.testing.assert_array_equal(loss.sample_counts, [3, 2])


    def test_sample_counts_round(self):
        loss = BatchedSubKernelSVCLoss(
            self.data, 
            self.labels, 
            sub_kernel_size=6
        )
        np.testing.assert_array_equal(loss.sample_counts, [4, 2])

    def test_sample_counts_clip(self):
        labels = np.array([0] * 9 + [1])

        loss = BatchedSubKernelSVCLoss(
            self.data, 
            labels, 
            sub_kernel_size=2
        )

        np.testing.assert_array_equal(loss.sample_counts, [1, 1])

    def test_batch_subkernels(self):
        loss = BatchedSubKernelSVCLoss(
            self.data,
            self.labels,
            sub_kernel_size=5,
            batch_size=2,
        )

        subkernels = loss._batch_subkernels()

        self.assertEqual(len(subkernels), 2)

        for data, labels in subkernels:
            self.assertEqual(len(data), 5)
            self.assertEqual(np.sum(labels == 0), 3)
            self.assertEqual(np.sum(labels == 1), 2)
            self.assertEqual(len(np.unique(data)), 5)


    def test_unique_sampling_without_replacement(self):
        loss = BatchedSubKernelSVCLoss(
            self.data,
            self.labels,
            sub_kernel_size=5,
            batch_size=2,
        )

        subkernels = loss._batch_subkernels()

        sampled = np.concatenate(
            [subkernel_data for subkernel_data, _ in subkernels]
        )

        self.assertEqual(len(np.unique(sampled, axis=0)), 10)

    def test_class_groups_reset_independently(self):
        loss = BatchedSubKernelSVCLoss(
            self.data,
            self.labels,
            sub_kernel_size=4,
            batch_size=2,
        )

        loss._batch_subkernels()

        self.assertEqual(len(loss.unused_idxs[0]), 2)
        self.assertEqual(len(loss.unused_idxs[1]), 0)

        loss.batch_size = 1
        loss._batch_subkernels()

        self.assertEqual(len(loss.unused_idxs[0]), 0)
        self.assertEqual(len(loss.unused_idxs[1]), 2)


    def test_batched_evaluate_returns_average_loss(self):
        loss = BatchedSubKernelSVCLoss(
            self.data,
            self.labels,
            sub_kernel_size=4,
            batch_size=2,
        )

        batch_1 = (self.data[:4], np.array([0, 0, 1, 1]))
        batch_2 = (self.data[4:8], np.array([0, 0, 1, 1]))

        loss._batch_subkernels = MagicMock(
            return_value=[batch_1, batch_2]
        )

        with patch.object(SVCLoss,"evaluate", side_effect=[1.0, 3.0]) as mock_evaluate:
            result = loss.evaluate(
                np.array([0.1, 0.2]),
                MagicMock(),
                self.data,
                self.labels,
            )

        self.assertEqual(result, 2.0)
        self.assertEqual(loss.loss_arr, [2.0])
        self.assertEqual(mock_evaluate.call_count, 2)



if __name__ == "__main__":
    unittest.main()