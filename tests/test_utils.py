import pathlib

import numpy as np
import pytest

from color_transfer_function_designer.lib.utils import (
    read_paraview_tf_from_json,
    read_slicer_tf_from_ascii,
    write_slicer_vp,
)


def _sample_luts():
    lut_rgb = np.array([[0.0, 1.0, 0.0, 0.0], [3.0, 0.5, 0.5, 0.5]], dtype=np.float64)
    lut_scalar_alpha = np.array([[0.0, 0.0], [3.0, 1.0]], dtype=np.float64)
    lut_gradient_alpha = np.array([[0.0, 0.0], [1.0, 0.8]], dtype=np.float64)
    return lut_rgb, lut_scalar_alpha, lut_gradient_alpha


@pytest.mark.parametrize("as_str", [False, True])
def test_write_slicer_vp_format(tmp_path, as_str):
    lut_rgb, lut_scalar_alpha, lut_gradient_alpha = _sample_luts()
    path = tmp_path / "tf.vp"
    write_slicer_vp(
        str(path) if as_str else path, lut_rgb, lut_scalar_alpha, lut_gradient_alpha
    )

    lines = path.read_text().splitlines()
    assert lines[:6] == ["1", "1", "1", "0.2", "0", "1"]
    # line 6: scalar opacities, prefixed by element count (2x2 = 4).
    assert lines[6].split()[0] == "4"
    assert lines[7].split()[0] == "4"  # gradient opacities
    assert lines[8].split()[0] == "8"  # rgb (2x4 = 8)


def test_read_slicer_tf_from_ascii_shapes():
    content = "\n".join(
        [
            "1",
            "1",
            "1",
            "0.2",
            "0",
            "1",
            "4 0.0 0.0 3.0 1.0",
            "4 0.0 0.0 1.0 0.8",
            "8 0.0 1.0 0.0 0.0 3.0 0.5 0.5 0.5",
        ]
    )
    colors, scalar_opacities, gradient_opacities = read_slicer_tf_from_ascii(content)
    assert colors.shape == (2, 4)
    assert scalar_opacities.shape == (2, 2)
    assert gradient_opacities.shape == (2, 2)
    np.testing.assert_allclose(colors[0], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(scalar_opacities[-1], [3.0, 1.0])


def test_vp_round_trip(tmp_path):
    lut_rgb, lut_scalar_alpha, lut_gradient_alpha = _sample_luts()
    path = tmp_path / "tf.vp"
    write_slicer_vp(path, lut_rgb, lut_scalar_alpha, lut_gradient_alpha)

    colors, scalar_opacities, gradient_opacities = read_slicer_tf_from_ascii(
        pathlib.Path(path).read_text()
    )
    np.testing.assert_allclose(colors, lut_rgb)
    np.testing.assert_allclose(scalar_opacities, lut_scalar_alpha)
    np.testing.assert_allclose(gradient_opacities, lut_gradient_alpha)


def test_read_paraview_tf_from_json_valid():
    data = [
        {
            "Points": [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0],
            "RGBPoints": [0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0],
        }
    ]
    colors, opacities, gradient = read_paraview_tf_from_json(data)
    assert colors.shape == (2, 4)
    assert opacities.shape == (2, 2)
    assert gradient is None
    np.testing.assert_allclose(opacities, [[0.0, 0.0], [3.0, 1.0]])


def test_read_paraview_tf_from_json_empty_raises():
    with pytest.raises(RuntimeError):
        read_paraview_tf_from_json([])
