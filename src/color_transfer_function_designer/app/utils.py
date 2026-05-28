import pathlib

import numpy as np


def write_slicer_vp(
    path: pathlib.Path | str,
    lut_rgb: np.ndarray,
    lut_scalar_alpha: np.ndarray,
    lut_gradient_alpha: np.ndarray,
) -> None:
    """
    Writes lut_rgb, lut_scalar_alpha, and lut_gradient_alpha to a Slicer3D .vp file.

    Parameters
    ----------
    path : destination file path
    lut_rgb : (N, 4) array [scalar, r, g, b]
    lut_scalar_alpha : (M, 2) array [scalar, alpha]
    lut_gradient_alpha : (M, 2) array [gradient, alpha]
    """
    lines = ["1", "1", "1", "0.2", "0", "1"]

    def _flat_line(arr: np.ndarray) -> str:
        flat = arr.flatten()
        return f"{len(flat)} " + " ".join(repr(float(v)) for v in flat)

    lines.append(_flat_line(lut_scalar_alpha))
    lines.append(_flat_line(lut_gradient_alpha))
    lines.append(_flat_line(lut_rgb))

    with pathlib.Path(path).open("w") as f:
        f.write("\n".join(lines) + "\n")


def read_slicer_tf_from_ascii(content: str):
    """
    Reads opacities (line 6), gradients (line 7), and colors (line 8)
    from the text passed in `content`.
    """
    lines = content.splitlines(keepends=False)
    work_lines = lines[6:]

    def parse_line(line_idx, skip_count, shape):
        row = np.fromstring(
            work_lines[line_idx].strip(),
            sep=" ",
            dtype=np.float64,
        )
        return row[skip_count:].reshape(shape)

    scalar_opacities = parse_line(0, 1, (-1, 2))  # [[scalar, opacity], ...]
    gradient_opacities = parse_line(1, 1, (-1, 2))  # [[grad, opacity], ...]
    colors = parse_line(2, 1, (-1, 4))  # [[scalar, r, g, b], ...]

    return colors, scalar_opacities, gradient_opacities


def read_paraview_tf_from_json(data):
    """
    Reads colors and scalar opacities from json. This method does not return gradient opacities.
    """
    if not len(data):
        e = RuntimeError()
        e.add_note("Invalid json. Cannot read paraview transfer function.")
        raise e
    opacities = np.array(data[0]["Points"], dtype=np.float64).reshape(
        -1, 4
    )  # value, opacity, midpoint, sharpness
    opacities = opacities[:, :2]  # drop midpoint, sharpness
    colors = np.array(data[0]["RGBPoints"], dtype=np.float64).reshape(
        -1, 4
    )  # value, r, g, b
    return colors, opacities, None
