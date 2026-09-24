from evaluation.coverage_counter import CoverageCounter, circle_intersects_cell


def make_counter(radius=0.36):
    counter = CoverageCounter.__new__(CoverageCounter)
    counter.env_min_x = -5.0
    counter.env_min_y = -5.0
    counter.grid_size = 1.0
    counter.num_cells_x = 10
    counter.num_cells_y = 10
    counter.robot_footprint_radius = radius
    return counter


def test_circle_cell_intersection_rejects_nearby_corner():
    assert circle_intersects_cell(0.0, 0.0, 0.36, 0.2, 0.2, 1.2, 1.2)
    assert not circle_intersects_cell(0.0, 0.0, 0.36, 0.3, 0.3, 1.3, 1.3)


def test_centered_rover_marks_only_containing_cell():
    assert set(make_counter()._footprint_cells(0.5, 0.5)) == {(5, 5)}


def test_footprint_crossing_cell_edge_marks_neighbor():
    assert set(make_counter()._footprint_cells(0.8, 0.5)) == {(5, 5), (6, 5)}


def test_footprint_is_clipped_to_environment():
    assert set(make_counter()._footprint_cells(-4.9, -4.9)) == {(0, 0)}
