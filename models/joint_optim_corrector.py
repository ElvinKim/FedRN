import numpy as np
import torch


class JointOptimCorrector:
    def __init__(self, queue_size, num_classes, data_size):
        self.queue_size = queue_size
        self.num_classes = num_classes

        self.id_2_data_index = {}
        self.counts = np.zeros(data_size, dtype=int)
        # probability histories of samples
        self.probability_history = torch.zeros(data_size, queue_size, num_classes)

        # labels
        self.hard_labels = torch.zeros(data_size, dtype=torch.int64)
        self.soft_labels = torch.zeros(data_size, num_classes)

    def get_data_indices(self, ids):
        data_indices = []

        next_index = max(self.id_2_data_index.values()) + 1 \
            if self.id_2_data_index\
            else 0

        for _id in ids:
            if _id in self.id_2_data_index:
                data_idx = self.id_2_data_index[_id]
            else:
                data_idx = next_index
                self.id_2_data_index[_id] = data_idx
                next_index += 1

            data_indices.append(data_idx)

        return data_indices

    def get_labels(self, ids, labels):
        data_indices = self.get_data_indices(ids)

        init_indices = np.where(self.counts[data_indices] == 0)[0]
        if len(init_indices):
            self.hard_labels[data_indices] = labels
            self.soft_labels[data_indices, labels] = 1.

        hard_labels = self.hard_labels[data_indices]
        soft_labels = self.soft_labels[data_indices]

        return hard_labels, soft_labels

    def update_probability_history(self, ids, probs):
        data_indices = self.get_data_indices(ids)

        curr_index = self.counts[data_indices] % self.queue_size
        self.probability_history[data_indices, curr_index] = probs
        self.counts[data_indices] += 1

    def update_labels(self):
        self.soft_labels = self.probability_history.mean(dim=1)
        self.hard_labels = torch.argmax(self.soft_labels, dim=1)

    def clear_data(self):
        self.probability_history.clear()
