from PIL import Image
import numpy as np


class ImageSignature(object):
    """
    Based on paper "AN IMAGE SIGNATURE FOR ANY KIND OF IMAGE" H. Chi Wong, Marshall Bern, and David Goldberg
    """
    def signature(self, image: Image, grid_size: int = 11) -> np.array:
        # Step 1: Convert to greyscale
        pixels = np.asarray(image.convert("L"))
        boundaries = self.crop(pixels)

        # Step 2: Define the grid
        x_coords = np.linspace(boundaries[0][0], boundaries[0][1], grid_size + 2, dtype=int)[1:-1]
        y_coords = np.linspace(boundaries[1][0], boundaries[1][1], grid_size + 2, dtype=int)[1:-1]

        # Step 3: compute the average gray level of the 􏰣 p * p
        avg_grey = self.average_gray(pixels, x_coords, y_coords)

        # Step 4: Compute array of differences for each grid point
        diff_mat = self.compute_diff(avg_grey)
        self.normalize(diff_mat)

        # Step 5: Flatten array and return signature
        return np.ravel(diff_mat).astype('int8')

    @staticmethod
    def average_gray(pixels: np.array, x_coords: np.array, y_coords: np.array) -> np.array:
        p = max([2.0, int(0.5 + min(pixels.shape) / 20.)])

        avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))

        for i, x in np.ndenumerate(x_coords):
            lower_x_lim = int(max([x - p / 2, 0]))
            upper_x_lim = int(min([lower_x_lim + p, pixels.shape[0]]))
            for j, y in np.ndenumerate(y_coords):
                lower_y_lim = int(max([y - p / 2, 0]))
                upper_y_lim = int(min([lower_y_lim + p, pixels.shape[1]]))

                avg_grey[i, j] = np.mean(pixels[lower_x_lim:upper_x_lim,
                                         lower_y_lim:upper_y_lim])

        return avg_grey

    @staticmethod
    def crop(pixels: np.array, ratio=(5, 95)) -> list:
        rw = np.cumsum(np.sum(np.abs(np.diff(pixels, axis=1)), axis=1))
        cw = np.cumsum(np.sum(np.abs(np.diff(pixels, axis=0)), axis=0))

        upper_column_limit = np.searchsorted(cw,
                                             np.percentile(cw, ratio[1]),
                                             side='left')
        lower_column_limit = np.searchsorted(cw,
                                             np.percentile(cw, ratio[0]),
                                             side='right')
        upper_row_limit = np.searchsorted(rw,
                                          np.percentile(rw, ratio[1]),
                                          side='left')
        lower_row_limit = np.searchsorted(rw,
                                          np.percentile(rw, ratio[0]),
                                          side='right')

        if lower_row_limit > upper_row_limit:
            lower_row_limit = int(ratio[0] / 100. * pixels.shape[0])
            upper_row_limit = int(ratio[1] / 100. * pixels.shape[0])
        if lower_column_limit > upper_column_limit:
            lower_column_limit = int(ratio[0] / 100. * pixels.shape[1])
            upper_column_limit = int(ratio[1] / 100. * pixels.shape[1])

        return [(lower_row_limit, upper_row_limit),
                (lower_column_limit, upper_column_limit)]

    @staticmethod
    def compute_diff(avg_grey) -> np.array:
        right_neighbors = -np.concatenate((np.diff(avg_grey),
                                           np.zeros(avg_grey.shape[0]).
                                           reshape((avg_grey.shape[0], 1))),
                                          axis=1)
        left_neighbors = -np.concatenate((right_neighbors[:, -1:],
                                          right_neighbors[:, :-1]),
                                         axis=1)

        down_neighbors = -np.concatenate((np.diff(avg_grey, axis=0),
                                          np.zeros(avg_grey.shape[1]).
                                          reshape((1, avg_grey.shape[1]))))

        up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))

        diagonals = np.arange(-avg_grey.shape[0] + 1,
                              avg_grey.shape[0])

        upper_left_neighbors = sum(
            [np.diagflat(np.insert(np.diff(np.diag(avg_grey, i)), 0, 0), i)
             for i in diagonals])
        lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:],
                                        (0, 1), mode='constant')

        flipped = np.fliplr(avg_grey)
        upper_right_neighbors = sum([np.diagflat(np.insert(
            np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
        lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:],
                                       (0, 1), mode='constant')

        return np.dstack(np.array([
            upper_left_neighbors,
            up_neighbors,
            np.fliplr(upper_right_neighbors),
            left_neighbors,
            right_neighbors,
            np.fliplr(lower_left_neighbors),
            down_neighbors,
            lower_right_neighbors]))

    @staticmethod
    def normalize(difference_array):
        mask = np.abs(difference_array) < 2 / 255.
        difference_array[mask] = 0.
        n_levels = 2

        if np.all(mask):
            return None

        positive_cutoffs = np.percentile(difference_array[difference_array > 0.],
                                         np.linspace(0, 100, n_levels + 1))
        negative_cutoffs = np.percentile(difference_array[difference_array < 0.],
                                         np.linspace(100, 0, n_levels + 1))

        for level, interval in enumerate([positive_cutoffs[i:i + 2]
                                          for i in range(positive_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array >= interval[0]) &
                             (difference_array <= interval[1])] = level + 1

        for level, interval in enumerate([negative_cutoffs[i:i + 2]
                                          for i in range(negative_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array <= interval[0]) &
                             (difference_array >= interval[1])] = -(level + 1)

        return None

    @staticmethod
    def normalized_distance(a, b):
        b = b.astype(int)
        a = a.astype(int)

        return np.linalg.norm(b - a) / (np.linalg.norm(b) + np.linalg.norm(a))