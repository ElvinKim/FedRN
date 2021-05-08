import numpy as np
import operator


class SelfieCorrector:
    def __init__(self, queue_size, uncertainty_threshold, noise_rate, num_classes):
        self.queue_size = queue_size
        self.threshold = uncertainty_threshold
        self.noise_rate = noise_rate

        # prediction histories of samples
        self.prediction_history = {}

        # Max Correctablility / predictive uncertainty
        self.max_certainty = -np.log(1.0 / float(num_classes))

        # Corrected label map
        self.corrected_labels = {}
        self.counts = {}

    def init_id_data(self, _id):
        if _id in self.prediction_history:
            # data for id already initialized
            return

        self.prediction_history[_id] = np.zeros(self.queue_size, dtype=int)
        self.counts[_id] = 0
        self.corrected_labels[_id] = -1

    def update_prediction_history(self, ids, outputs):
        for _id, output in zip(ids, outputs):
            self.init_id_data(_id)

            # append the predicted label to prediction history
            pred_label = np.argmax(output)
            curr_index = self.counts[_id] % self.queue_size
            self.prediction_history[_id][curr_index] = pred_label
            self.counts[_id] += 1

    def separate_clean_and_unclean_samples(self, ids, loss_array):
        num_clean_instances = int(np.ceil(float(len(ids)) * (1.0 - self.noise_rate)))

        loss_map = {_id: loss for _id, loss in zip(ids, loss_array)}
        loss_map = dict(sorted(loss_map.items(), key=operator.itemgetter(1), reverse=False))

        clean_batch = list(loss_map.keys())[:num_clean_instances]

        return clean_batch

    def correct_samples(self, ids):
        # check correctability for each sample
        corrected_batch = []
        for _id in ids:
            pred_label_history = self.prediction_history[_id]
            pred_label_2_counts = {}

            for pred_label in pred_label_history:
                if pred_label not in pred_label_2_counts:
                    pred_label_2_counts[pred_label] = 1
                else:
                    pred_label_2_counts[pred_label] += 1

            negative_entropy = 0.0
            mode_label = None    # most common label
            for pred_label, count in pred_label_2_counts.items():
                prob = float(count) / float(self.queue_size)
                negative_entropy += prob * np.log(prob)

                # Update mode label
                if mode_label is None:
                    mode_label = pred_label
                elif count > pred_label_2_counts[mode_label]:
                    mode_label = pred_label

            certainty = -1.0 * negative_entropy / self.max_certainty

            if certainty <= self.threshold:
                self.corrected_labels[_id] = mode_label

            if self.corrected_labels[_id] != -1:
                corrected_batch.append(_id)

        return corrected_batch

    def correct_and_select_certain_samples(self, ids, X, y, clean_batch, corrected_batch):
        id_2_index = {_id: i for i, _id in enumerate(ids)}

        # Correct samples
        for _id in corrected_batch:
            y[id_2_index[_id]] = int(self.corrected_labels[_id])

        # Select high certainty samples
        high_certainty_samples = set(clean_batch) | set(corrected_batch)
        keep_indices = [id_2_index[_id] for _id in high_certainty_samples]

        X = X[keep_indices]
        y = y[keep_indices]

        return X, y

    def patch_clean_with_corrected_sample_batch(self, ids, X, y, loss_array):
        # 1. separate clean and unclean samples
        clean_batch = self.separate_clean_and_unclean_samples(ids, loss_array)
        # 2. get corrected samples
        corrected_batch = self.correct_samples(ids)
        # 3. merging
        X, y = self.correct_and_select_certain_samples(ids, X, y, clean_batch, corrected_batch)
        return X, y

    def clear_predictions(self):
        self.prediction_history.clear()
        self.counts.clear()
