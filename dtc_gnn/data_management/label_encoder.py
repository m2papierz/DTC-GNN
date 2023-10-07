import torch


class LabelEncoder:
    def __init__(self):
        self._labels = ["I", "X", "Z", "Y"]
        self._one_hots = torch.eye(
            len(self._labels), len(self._labels),
            dtype=torch.float32
        )

    @property
    def label_to_one_hot_dict(self):
        return {
            label: encoding for label, encoding in
            zip(sorted(self._labels), [t for t in self._one_hots])
        }

    @property
    def one_hot_to_label_dict(self):
        return {
            str(v): k for k, v in self.label_to_one_hot_dict.items()
        }

    def encode(self, label):
        return self.label_to_one_hot_dict[label]

    def decode(self, one_hot):
        return self.one_hot_to_label_dict[str(one_hot)]
