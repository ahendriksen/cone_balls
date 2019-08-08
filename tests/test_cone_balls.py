#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cone_balls` package."""


import unittest
import cone_balls.cone_balls as cb
import pyqtgraph as pq
import astra
import numpy as np
import torch

class TestCone_balls(unittest.TestCase):
    """Tests for `cone_balls` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_move_up(self):
        """Test something."""
        interactive = True
        pixel_size = 2.4e-3
        det_row_count = det_col_count = 500
        num_angles = 1
        sod = 10.0
        sdd = 10.0

        pg = cb.generate_cone_pg(
            (pixel_size, pixel_size), (det_row_count, det_col_count),
            num_angles, sod, sdd
        )
        ball_pos, ball_radius = cb.generate_balls(1, 1)
        ball_pos[:] = 0.0
        ball_radius[:] = 0.01

        proj_data = []
        for z in np.linspace(0, 0.5, 10):
            pg_moved = cb.move_detector(pg, z)
            pg_moved = cb.move_source(pg_moved, z)

            proj_data.append(
                list(cb.generate_projections(pg_moved, ball_pos, ball_radius))[0]
            )

        proj_data = np.array(proj_data)

        if interactive:
            app = pq.mkQApp()
            # Mark corner of 10x10 pixels, so you can check that
            # pyqtgraph puts them in the top-left corner.
            proj_data[:, :10, :10] = -1
            pq.image(proj_data, axes={"t": 0, "x": 2, "y": 1})
            app.exec_()

    def test_parallel_move(self):
        interactive = True
        pixel_size = 2.4e-3
        det_row_count = det_col_count = 500
        num_angles = 1

        pg = cb.generate_parallel_pg(
            (pixel_size, pixel_size), (det_row_count, det_col_count),
            num_angles
        )
        ball_pos, ball_radius = cb.generate_balls(1, 1)
        ball_pos[:] = 0.0
        ball_radius[:] = 0.01

        proj_data = []
        for z in np.linspace(0, 0.5, 10):
            proj_data.append(
                list(cb.generate_projections(pg, ball_pos + z, ball_radius, cone=False))[0]
            )

        proj_data = np.array(proj_data)

        if interactive:
            app = pq.mkQApp()
            # Mark corner of 10x10 pixels, so you can check that
            # pyqtgraph puts them in the top-left corner.
            proj_data[:, :10, :10] = -1e-3
            pq.image(proj_data, axes={"t": 0, "x": 2, "y": 1})
            app.exec_()

    def test_parallel(self):
        interactive = True
        pixel_size = 2.4e-3
        det_row_count = det_col_count = 500
        num_angles = 100

        pg = cb.generate_parallel_pg(
            (pixel_size, pixel_size), (det_row_count, det_col_count),
            num_angles
        )
        ball_pos, ball_radius = cb.generate_balls(100, 1)

        proj_data = np.array(list(cb.generate_projections(pg, ball_pos, ball_radius, cone=False)))

        if interactive:
            app = pq.mkQApp()
            # Mark corner of 10x10 pixels, so you can check that
            # pyqtgraph puts them in the top-left corner.
            proj_data[:, :10, :10] = -1e-3
            pq.image(proj_data, axes={"t": 0, "x": 2, "y": 1})
            app.exec_()

    def test_volume(self):

        ball_pos, ball_radius = cb.generate_balls(100, 1)

        ball_pos = torch.tensor(
            [[-1, 0, 0],
             [0, 0, 0],
             [0, 0.2, 1],
             [-1, -1, -1],
            ],
            dtype=torch.float32,
            device=torch.device('cuda'),
        )
        ball_radius = torch.tensor(
            [0.2, 0.2, 0.2, 0.2],
            dtype=torch.float32,
            device=torch.device('cuda'),
        )
        size = torch.tensor((2, 2, 2), dtype=torch.float32)
        shape = (201, 201, 201)
        voxel_size = size / torch.tensor(shape, dtype=torch.float32)
        lower_left_voxel_center = - size / 2 + voxel_size / 2
        print(voxel_size)
        print(lower_left_voxel_center)
        print(lower_left_voxel_center - voxel_size / 2)

        import cone_balls_cuda as cbc

        out_volume = torch.zeros(*shape, dtype=torch.float32).cuda()
        cbc.compute_volume(
            lower_left_voxel_center.cuda(),
            voxel_size.cuda(),
            ball_pos.cuda(),
            ball_radius.cuda(),
            out_volume,
            super_sampling=8
        )
        app = pq.mkQApp()
        pq.image(
            out_volume.detach().cpu().numpy(),
            scale=(1, -1),
            axes=dict(zip("tyx", range(3))),
        )
        app.exec_()
