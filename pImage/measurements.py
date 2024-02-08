import cv2, numpy as np

from .transformations import flat_field_correction


class SiftComparison:

    def __init__(
        self,
        vector,
        reference_points=None,
        measured_points=None,
        reference_keypoints=None,
        measured_keypoints=None,
        match_mask=None,
        reference_image=None,
        measured_image=None,
    ):

        super().__init__()
        self.vector = vector
        self.reference_points = reference_points
        self.measured_points = measured_points
        self.reference_keypoints = reference_keypoints
        self.measured_keypoints = measured_keypoints
        self.match_mask = match_mask
        self.input_array_shape = reference_image.shape
        self.reference_image = reference_image
        self.measured_image = measured_image

    @staticmethod
    def from_images(reference_image, measured_image, max_pt_num=25):
        import cv2

        sift = cv2.SIFT_create()

        reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)
        measured_keypoints, measured_descriptors = sift.detectAndCompute(measured_image, None)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(reference_descriptors, measured_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < max_pt_num:
            max_pt_num = len(matches)

        reference_points = np.expand_dims(
            np.array([reference_keypoints[matches[i].queryIdx].pt for i in range(max_pt_num)]), axis=1
        ).astype(np.float32)
        measured_points = np.expand_dims(
            np.array([measured_keypoints[matches[i].trainIdx].pt for i in range(max_pt_num)]), axis=1
        ).astype(np.float32)

        vector, match_mask = cv2.estimateAffinePartial2D(reference_points, measured_points)

        return SiftComparison(
            vector,
            reference_points,
            measured_points,
            reference_keypoints,
            measured_keypoints,
            match_mask,
            reference_image,
            measured_image,
        )

    @property
    def translation_y(self):
        return self.vector[1, 2] - self._offset_vector[1]

    @property
    def translation_x(self):
        return self.vector[0, 2] - self._offset_vector[0]

    @property
    def rotation(self):
        return -np.degrees(np.arctan2(self.vector[1, 0], self.vector[0, 0]))

    @property
    def _offset_vector(self):
        # to obtainrotation matrix with offset for the center :
        #   cv2.getRotationMatrix2D(center, angle_deg(positive=ccw), 1.0)
        # to obtain new values with current ones : np.matmul(self.vector,np.array([[40],[40],[1]]))
        center_origin = np.array(self.input_array_shape[0:2]).astype(np.float64) / 2
        theta = np.radians(self.rotation)
        x = center_origin[0] * (1 - np.cos(theta)) - center_origin[1] * np.sin(theta)
        y = center_origin[1] * (1 - np.cos(theta)) + center_origin[0] * np.sin(theta)
        return x, y

    @property
    def trans_vector(self):
        return self.translation_x, self.translation_y, self.rotation

    # def raw_pts(self, point, reshaped=True):
    #     if reshaped:
    #         if point == 1:
    #             return self._reshape_points(self.reference_points) if self.reference_points is not None else None
    #         if point == 2:
    #             return self._reshape_points(self.measured_points) if self.measured_points is not None else None
    #     else:
    #         return self.reference_points if point == 1 else self.measured_points

    # def _reshape_points(self, points):
    #     return np.reshape(points, (points.shape[0], 2))

    # def kept_pts(self, point, reshaped=True):
    #    return kept_points(self.raw_pts(point, reshaped), self.match_mask)

    def draw_keypoints(self, which="reference", ax=None):
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()

        if which == "reference":
            keypoints = self.reference_keypoints
            image = self.reference_image
        elif which == "measured":
            keypoints = self.measured_keypoints
            image = self.measured_image
        else:
            raise ValueError

        ax.imshow(cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        return ax

    def __repr__(self):
        try:
            return self.__str__()
        except Exception:
            return "Sift " + self.vector.__str__()

    def __str__(self):
        _str = "Sift "
        for index, item in enumerate(["x", "y", "θ"]):
            _str += item + f" : {self.trans_vector[index]:.2f}" + " "
        return _str


def correlation_2D(image1_gray, image2_gray, flat_field=None, flat_field_corrected=False, gaussian=None):
    """Calculate the 2D correlation image between two images

    Args:
        image1_gray (_type_): _description_
        image2_gray (_type_): _description_

    Returns:
        _type_: shifts, (x,y), maximum_correlation_point (x, y)
    """
    # https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
    from scipy.signal import fftconvolve

    if flat_field_corrected:
        image1_gray = flat_field_correction(image1_gray, gaussian=gaussian, flat_field=flat_field)
        image2_gray = flat_field_correction(image2_gray, gaussian=gaussian, flat_field=flat_field)

    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    image1_gray = image1_gray.astype("float")
    image2_gray = image2_gray.astype("float")

    # get rid of the averages, otherwise the results are not good
    image1_gray -= np.mean(image1_gray)
    image2_gray -= np.mean(image2_gray)

    corr_map = fftconvolve(image1_gray, image2_gray[::-1, ::-1], mode="same")
    correlation_center = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    shifts = np.flip(correlation_center - (np.array(corr_map.shape) / 2))
    correlation_center = np.flip(correlation_center)
    # calculate the correlation image; note the flipping of onw of the images

    return {
        "correlation_image": corr_map,
        "shifts": shifts,
        "maximum_correlation_point": correlation_center,
        "reference_image": image1_gray,
        "measured_image": image2_gray,
    }
