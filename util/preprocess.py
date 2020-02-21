import numpy as np
import cv2
import util.gaze


def preprocess_unityeyes_image(img, json_data, oh=90, ow=150, heatmap_h=45, heatmap_w=75):
    # Prepare to segment eye image
    ih, iw = img.shape[:2]
    ih_2, iw_2 = ih/2.0, iw/2.0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # Normalize to eye width.
    scale = ow/eye_width

    transform = np.zeros((2, 3))
    transform[0, 2] = -eye_middle[0] * scale + 0.5 * ow
    transform[1, 2] = -eye_middle[1] * scale + 0.5 * oh
    transform[0, 0] = scale
    transform[1, 1] = scale
    
    transform_inv = np.zeros((2, 3))
    transform_inv[:, 2] = -transform[:, 2]
    transform_inv[0, 0] = 1/scale
    transform_inv[1, 1] = 1/scale
    
    # Apply transforms
    eye = cv2.warpAffine(img, transform, (ow, oh))

    # Normalize eye image
    eye = cv2.equalizeHist(eye)
    eye = eye.astype(np.float32)
    eye = eye / 255.0

    # Gaze
    # Convert look vector to gaze direction in polar angles
    look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3]
    look_vec[0] = -look_vec[0]
    original_gaze = util.gaze.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
    gaze = util.gaze.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
    if gaze[1] > 0.0:
        gaze[1] = np.pi - gaze[1]
    elif gaze[1] < 0.0:
        gaze[1] = -(np.pi + gaze[1])
    gaze = gaze.astype(np.float32)

    iris_center = np.mean(iris_landmarks[:, :2], axis=0)

    landmarks = np.concatenate([interior_landmarks[:, :2],  # 8
                                iris_landmarks[::2, :2],  # 8
                                iris_center.reshape((1, 2)),
                                [[iw_2, ih_2]],  # Eyeball center
                                ])  # 18 in total

    landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1))
    landmarks = np.asarray(landmarks * transform.T) * np.array([heatmap_w/ow, heatmap_h/oh])
    landmarks = landmarks.astype(np.float32)

    # Swap columns so that landmarks are in (y, x), not (x, y)
    # This is because the network outputs landmarks as (y, x) values.
    temp = np.zeros((34, 2), dtype=np.float32)
    temp[:, 0] = landmarks[:, 1]
    temp[:, 1] = landmarks[:, 0]
    landmarks = temp

    heatmaps = get_heatmaps(w=heatmap_w, h=heatmap_h, landmarks=landmarks)

    assert heatmaps.shape == (34, heatmap_h, heatmap_w)

    return {
        'img': eye,
        'transform': transform,
        'transform_inv': transform_inv,
        'eye_middle': eye_middle,
        'heatmaps': heatmaps,
        'landmarks': landmarks,
        'gaze': gaze
    }


def gaussian_2d(w, h, cx, cy, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    xs, ys = np.meshgrid(
        np.linspace(0, w - 1, w, dtype=np.float32),
        np.linspace(0, h - 1, h, dtype=np.float32)
    )

    assert xs.shape == (h, w)
    alpha = -0.5 / (sigma ** 2)
    heatmap = np.exp(alpha * ((xs - cx) ** 2 + (ys - cy) ** 2))
    return heatmap


def get_heatmaps(w, h, landmarks):
    heatmaps = []
    for (y, x) in landmarks:
        heatmaps.append(gaussian_2d(w, h, cx=x, cy=y, sigma=3.0))
    return np.array(heatmaps)
