import pytest

from datasets.unity_eyes import UnityEyesDataset


def test_unity_eyes():
    ds = UnityEyesDataset()
    sample = ds[0]
    assert sample['full_img'].shape == (600, 800, 3)
    assert sample['img'].shape == (90, 150, 3)
    assert float(sample['json_data']['eye_details']['iris_size']) == 0.9349335
