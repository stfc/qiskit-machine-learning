import numpy as np

class DataBatcher:
    """
    A class used to batch dataset and labels.
    """

    def __init__(self, dataset, labels):
        """
        Initialize a DataBatches object with the input dataset and corresponding labels.

        Args:
            dataset (numpy array): A numpy array of shape (num_samples, num_features) containing the input dataset.
            labels (numpy array): A numpy array of shape (num_samples,) containing the corresponding labels for the dataset.

        Returns:
            None
        """
        self.dataset = dataset
        self.labels = labels
        self.num_samples = len(dataset)
        self.unique_labels, self.label_counts = np.unique(labels, return_counts=True)

    def balanced_batches(self, batch_size, shuffle=False):
        """
        Generate a list of balanced batches, where each batch contains the same number of samples from each label.

        Args:
            batch_size (int): The desired size of each batch.
            shuffle (bool): if True, shuffle batches.

        Returns:
            batches (List): a list of batches where each batch is a tuple containing the batch data and corresponding labels.
        """
        if batch_size > self.num_samples:
            raise ValueError(
                f"Batch size {batch_size} is larger than the dataset size {self.num_samples}"
            )
        if batch_size > 2 * np.min(self.label_counts):
            raise ValueError(
                f"Batch size {batch_size} is 2x larger than the smallest label size {np.min(self.label_counts)}"
            )
        samples_per_label = batch_size // len(self.unique_labels)
        batches = []
        for _ in range(self.num_samples // batch_size):
            batch_data = []
            batch_labels = []
            for l in self.unique_labels:
                label_indices = np.where(self.labels == l)[0]
                if shuffle:
                    np.random.shuffle(label_indices)
                if samples_per_label > len(label_indices):
                    batch_indices = label_indices
                else:
                    batch_indices = label_indices[:samples_per_label]
                batch_data.append(self.dataset[batch_indices])
                batch_labels.append(self.labels[batch_indices])
            batch_data = np.concatenate(batch_data, axis=0)
            batch_labels = np.concatenate(batch_labels, axis=0)
            batches.append((batch_data, batch_labels))

        return batches

    def imbalanced_batches(self, batch_size, keep_ratio=False, shuffle=False):
        """
        Generate a list of imbalanced batches, where each batch may contain a different number of samples from each label.

        Args:
            batch_size (int): The desired size of each batch.
            keep_ratio (bool): If True, maintain the same relative frequency of each label as in the original dataset.
                            If False, use the absolute frequency of each label to determine the number of samples per label.
            shuffle (bool): If True, shuffle batches.

        Returns:
            batches (list): a list of batches where each batch is a tuple containing the batch data and corresponding labels.
        """
        if batch_size > self.num_samples:
            raise ValueError(
                f"Batch size {batch_size} is larger than the dataset size {self.num_samples}"
            )
        if keep_ratio:
            # calculate the number of samples per label based on the relative label frequencies
            label_freqs = self.label_counts / np.sum(self.label_counts)
            samples_per_label = np.round(batch_size * label_freqs).astype(int)
        else:
            # calculate the number of samples per label based on the absolute label frequencies
            samples_per_label = np.round(
                batch_size * self.label_counts / np.sum(self.label_counts)
            ).astype(int)
        batches = []
        for _ in range(self.num_samples // batch_size):
            batch_data = []
            batch_labels = []
            for l, num_samples in zip(self.unique_labels, samples_per_label):
                label_indices = np.where(self.labels == l)[0]
                if shuffle:
                    np.random.shuffle(label_indices)
                batch_indices = label_indices[:num_samples]
                batch_data.append(self.dataset[batch_indices])
                batch_labels.append(self.labels[batch_indices])
            batch_data = np.concatenate(batch_data, axis=0)
            batch_labels = np.concatenate(batch_labels, axis=0)
            batches.append((batch_data, batch_labels))

        return batches