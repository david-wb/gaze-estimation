import numpy as np
import cv2


def preprocess_unityeyes_image(img, json_data, oh=90, ow=150):
    # Prepare to segment eye image
    ih, iw = img.shape[:2]

    def process_coords(coords_list):
        coords = [eval(l) for l in coords_list]
        return np.array([(x, ih-y, z) for (x, y, z) in coords])
    
    interior_landmarks = process_coords(json_data['interior_margin_2d'])
    caruncle_landmarks = process_coords(json_data['caruncle_2d'])
    iris_landmarks = process_coords(json_data['iris_2d'])

    left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
    right_corner = interior_landmarks[8, :2]
    eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
    eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                          np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

    scale = ow/eye_width
    original_eyeball_radius = 71.7593
    eyeball_radius = original_eyeball_radius * scale  # See: https://goo.gl/ZnXgDE
    radius = np.float32(eyeball_radius)

    transform = np.zeros((2, 3))
    transform[0, 2] = -eye_middle[0] * scale + 0.5 * ow # * scale_inv
    transform[1, 2] = -eye_middle[1] * scale + 0.5 * oh# * scale_inv
    transform[0, 0] = scale
    transform[1, 1] = scale
    
    transform_inv = np.zeros((2, 3))
    transform_inv[:, 2] = -transform[:, 2]
    transform_inv[0, 0] = 1/scale
    transform_inv[1, 1] = 1/scale
    
    # Apply transforms
    eye = cv2.warpAffine(img, transform, (ow, oh))

    # get heatmaps
    def gaussian_2d(shape, centre, sigma=1.0):
        """Generate heatmap with single 2D gaussian."""
        xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
        ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
        alpha = -0.5 / (sigma ** 2)
        heatmap = np.exp(alpha * ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2))
        return heatmap

    heatmaps = get_heatmaps((oh, ow), iris_landmarks, transform)
    heatmaps = [cv2.resize(x, (75, 45)) for x in heatmaps]

    return {
        'img': eye,
        'transform': transform,
        'transform_inv': transform_inv,
        'radius': radius,
        'original_radius': original_eyeball_radius,
        'eye_middle': eye_middle,
        'heatmaps': heatmaps
    }


def get_heatmaps(shape, iris_landmarks, transform):

    def gaussian_2d(shape, centre, sigma=1.0):
        """Generate heatmap with single 2D gaussian."""
        xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
        ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
        alpha = -0.5 / (sigma ** 2)
        heatmap = np.exp(alpha * ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2))
        return heatmap

    heatmaps = []
    for (x, y, z) in iris_landmarks:
        x, y = np.matmul(transform, [x, y, 1.0])
        heatmaps.append(gaussian_2d(shape, (int(x), int(y)), sigma=5.0))
    return heatmaps
