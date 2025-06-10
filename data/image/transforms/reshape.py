


class Flatten():
    def __init__(self, batch_dim=False):
        self.batch_dim = batch_dim

    def __call__(self, x):
        if self.batch_dim:
            return x.reshape((x.shape[0], -1))
        else:
            return x.reshape(-1)