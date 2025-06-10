

class Select():
    def __init__(self, key):
        self.key = key

    def __call__(self, x):
        return x[self.key]