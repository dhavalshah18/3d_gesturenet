class Normalize(object):
    """Normalizes keypoints.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        assert len(tensor.size()) == 4

        return tensor.sub_(self.mean).div_(self.std)

