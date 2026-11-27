"""The occupancy grid renderer: the simulator's palette and geometry on a slam_toolbox grid."""

import numpy as np

from gpt_pet_mcp import config
from gpt_pet_mcp.maprender import FREE_WHITE, OCCUPIED_BLACK, UNKNOWN_GREY, MapOverlays, OccupancyGridRenderer, classify


def _grid(width=20, height=20, fill=0):
    return np.full((height, width), fill, dtype=np.int8)


def test_classify_uses_the_ros_map_palette():
    data = np.array([[-1, 0, 49, 50, 100]], dtype=np.int8)
    assert classify(data).tolist() == [[UNKNOWN_GREY, FREE_WHITE, FREE_WHITE, OCCUPIED_BLACK, OCCUPIED_BLACK]]


def test_render_before_any_map_is_the_grey_placeholder():
    img = OccupancyGridRenderer().render()
    assert img.size == (100, 100)
    assert img.getpixel((50, 50)) == (UNKNOWN_GREY,) * 3


def test_cells_are_five_pixels_with_x_right_and_y_down():
    renderer = OccupancyGridRenderer()
    data = _grid()
    data[2, 3] = 100  # row 2 (+y), column 3 (+x)
    renderer.update(20, 20, 0.05, 1.0, -2.0, data.ravel())
    img = renderer.render()
    assert img.size == (100, 100)  # 20 cells x 5 px, fully known so no crop
    assert img.getpixel((3 * 5 + 2, 2 * 5 + 2)) == (OCCUPIED_BLACK,) * 3
    assert img.getpixel((0, 0)) == (FREE_WHITE,) * 3
    assert img.getpixel((2, 2 * 5 + 2)) == (FREE_WHITE,) * 3


def test_overlays_use_the_grid_origin():
    renderer = OccupancyGridRenderer()
    renderer.update(20, 20, 0.05, 1.0, -2.0, _grid().ravel())
    # the robot half a metre into the grid along both axes: pixel (50, 50)
    overlays = MapOverlays(agent_position={"x": 1.5, "y": 0.0, "z": -1.5}, agent_yaw_deg=0.0)
    img = renderer.render(overlays)
    reds = [
        (x, y)
        for x in range(100)
        for y in range(100)
        if img.getpixel((x, y)) == config.AGENT_COLOR
    ]
    assert reds, "the robot triangle was not drawn"
    xs, ys = zip(*reds)
    assert 40 <= sum(xs) / len(xs) <= 60 and 40 <= sum(ys) / len(ys) <= 60
    # yaw 0 faces +z, which is down the image: the tip (farthest red pixel) lies below the centroid
    assert max(ys) - 50 > 50 - min(ys)


def test_crop_to_known_cells_keeps_world_coordinates():
    renderer = OccupancyGridRenderer()
    data = _grid(width=100, height=100, fill=-1)
    data[40:60, 40:60] = 0  # a 1 m square of known floor in the middle
    data[50, 50] = 100
    renderer.update(100, 100, 0.05, 0.0, 0.0, data.ravel())
    img = renderer.render()
    # 20 known cells + 10 cells of margin each side = 40 cells = 200 px
    assert img.size == (200, 200)
    # the occupied cell at world (2.5, 2.5) -> crop origin (1.5, 1.5) -> 1 m in = 100 px
    assert img.getpixel((102, 102)) == (OCCUPIED_BLACK,) * 3
    assert img.getpixel((2, 2)) == (UNKNOWN_GREY,) * 3


def test_free_cells_are_cell_centres():
    renderer = OccupancyGridRenderer()
    data = _grid(width=4, height=3, fill=100)
    data[1, 2] = 0
    renderer.update(4, 3, 0.5, 10.0, 20.0, data.ravel())
    cells = renderer.free_cells_xz()
    assert cells.shape == (1, 2)
    assert cells[0].tolist() == [10.0 + 2.5 * 0.5, 20.0 + 1.5 * 0.5]
    assert OccupancyGridRenderer().free_cells_xz() is None
