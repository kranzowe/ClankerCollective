"""Unit tests for load_occupancy_pointcloud (PGM + YAML parsing)."""
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from og_estimation_localization.core import load_occupancy_pointcloud


def write_map_pair(tmpdir, pixels_uint8, resolution=0.1, origin=(0.0, 0.0, 0.0),
                   occupied_thresh=0.65, negate=0):
    """Helper: write a synthetic <name>.pgm + <name>.yaml pair."""
    pgm_path = os.path.join(tmpdir, 'map.pgm')
    yaml_path = os.path.join(tmpdir, 'map.yaml')
    Image.fromarray(pixels_uint8, mode='L').save(pgm_path)
    with open(yaml_path, 'w') as f:
        f.write(
            f"image: map.pgm\n"
            f"resolution: {resolution}\n"
            f"origin: [{origin[0]}, {origin[1]}, {origin[2]}]\n"
            f"negate: {negate}\n"
            f"occupied_thresh: {occupied_thresh}\n"
            f"free_thresh: 0.25\n"
        )
    return yaml_path


class TestMapLoaderBasics:
    def test_single_black_pixel(self, tmp_path):
        # 3x3 map: only the bottom-left pixel is occupied (black).
        # Image rows go top-down; image[2, 0] is the bottom-left in image coords,
        # which corresponds to map origin (0, 0).
        img = np.full((3, 3), 255, dtype=np.uint8)
        img[2, 0] = 0  # black -> occupied
        yaml_path = write_map_pair(str(tmp_path), img, resolution=1.0,
                                   origin=(0.0, 0.0, 0.0))
        pts = load_occupancy_pointcloud(yaml_path)
        assert pts.shape == (1, 2)
        np.testing.assert_array_almost_equal(pts[0], [0.0, 0.0])

    def test_top_left_pixel_maps_to_top_left_of_map(self, tmp_path):
        # 3x3 map, only top-left pixel (image[0,0]) occupied.
        # Bottom-left of map is origin (0,0); top-left of a 3x3 grid w/ res 1 is (0, 2).
        img = np.full((3, 3), 255, dtype=np.uint8)
        img[0, 0] = 0
        yaml_path = write_map_pair(str(tmp_path), img, resolution=1.0,
                                   origin=(0.0, 0.0, 0.0))
        pts = load_occupancy_pointcloud(yaml_path)
        assert pts.shape == (1, 2)
        np.testing.assert_array_almost_equal(pts[0], [0.0, 2.0])

    def test_origin_offset_applied(self, tmp_path):
        img = np.full((3, 3), 255, dtype=np.uint8)
        img[2, 0] = 0  # bottom-left
        yaml_path = write_map_pair(str(tmp_path), img, resolution=0.5,
                                   origin=(-10.0, 5.0, 0.0))
        pts = load_occupancy_pointcloud(yaml_path)
        np.testing.assert_array_almost_equal(pts[0], [-10.0, 5.0])

    def test_resolution_scales_coordinates(self, tmp_path):
        img = np.full((3, 3), 255, dtype=np.uint8)
        img[2, 2] = 0  # bottom-right
        yaml_path = write_map_pair(str(tmp_path), img, resolution=0.5,
                                   origin=(0.0, 0.0, 0.0))
        pts = load_occupancy_pointcloud(yaml_path)
        # Bottom-right pixel (j=2) should map to x = 2 * 0.5 = 1.0.
        np.testing.assert_array_almost_equal(pts[0], [1.0, 0.0])

    def test_white_pixels_excluded(self, tmp_path):
        img = np.full((5, 5), 255, dtype=np.uint8)  # all white = all free
        yaml_path = write_map_pair(str(tmp_path), img)
        pts = load_occupancy_pointcloud(yaml_path)
        assert pts.shape[0] == 0

    def test_threshold_applied(self, tmp_path):
        # Mid-gray pixel: prob = (255-128)/255 ≈ 0.498 -> below 0.65 threshold.
        img = np.full((3, 3), 128, dtype=np.uint8)
        yaml_path = write_map_pair(str(tmp_path), img, occupied_thresh=0.65)
        pts = load_occupancy_pointcloud(yaml_path)
        assert pts.shape[0] == 0
        # With a lower threshold, those same pixels should now count as occupied.
        yaml_path2 = write_map_pair(str(tmp_path / 'sub') if False else str(tmp_path),
                                    img, occupied_thresh=0.3)
        pts2 = load_occupancy_pointcloud(yaml_path2)
        assert pts2.shape[0] == 9


class TestFullRoom:
    def test_outline_of_room(self, tmp_path):
        # 5x5 grid, walls = black perimeter.
        img = np.full((5, 5), 255, dtype=np.uint8)
        img[0, :] = 0
        img[-1, :] = 0
        img[:, 0] = 0
        img[:, -1] = 0
        yaml_path = write_map_pair(str(tmp_path), img, resolution=1.0,
                                   origin=(0.0, 0.0, 0.0))
        pts = load_occupancy_pointcloud(yaml_path)
        # 16 perimeter pixels (5+5+3+3 unique).
        assert pts.shape[0] == 16
        # All points lie on the box [0,4] x [0,4].
        assert pts[:, 0].min() == 0.0
        assert pts[:, 0].max() == 4.0
        assert pts[:, 1].min() == 0.0
        assert pts[:, 1].max() == 4.0
