# -*- coding: utf-8 -*-

"""Main module."""
import click
from tqdm import tqdm
import numpy as np
import astra
import torch
import cone_balls_cuda as cb
import pyqtgraph as pq
import tifffile
import os
import logging
import warnings
import pickle

torch_options = {"dtype": torch.float32, "device": torch.device(type="cuda")}
BATCH = 500


def generate_projection_geometry(det_spacing, det_shape, num_angles, SOD, SDD):
    angles = np.linspace(0, 2 * np.pi, num_angles, False)
    pg = astra.create_proj_geom(
        "cone", *det_spacing, *det_shape, angles, SOD, SDD - SOD
    )
    return astra.geom_2vec(pg)


def generate_balls(num_balls, pos_limit):
    ball_pos = (0.5 - torch.rand(num_balls, 3, **torch_options)) * 2.0 * pos_limit
    ball_radius = torch.rand(num_balls, **torch_options) * pos_limit / 10
    return (ball_pos, ball_radius)


def generate_projections(pg, ball_pos, ball_radius):
    # Get source position, detector position, detector u position, and
    # detector v position in Z, Y, X format ([:, ::-1]).
    ray_pos = pg["Vectors"][:, 0:3][:, ::-1]
    det_pos = pg["Vectors"][:, 3:6][:, ::-1]
    det_u = pg["Vectors"][:, 6:9][:, ::-1]
    det_v = pg["Vectors"][:, 9:12][:, ::-1]
    det_shape = (pg["DetectorRowCount"], pg["DetectorColCount"])

    ray_pos = torch.tensor(np.ascontiguousarray(ray_pos), **torch_options)
    det_pos = torch.tensor(np.ascontiguousarray(det_pos), **torch_options)
    det_u = torch.tensor(np.ascontiguousarray(det_u), **torch_options)
    det_v = torch.tensor(np.ascontiguousarray(det_v), **torch_options)

    num_angles = len(ray_pos)

    for i in range(0, num_angles, BATCH):
        # Generate BATCH number of projection images (or n, if there
        # are fewer remaining angles)
        n = min(BATCH, num_angles - i)
        proj_data = torch.zeros(n, *det_shape, **torch_options)
        proj_data.zero_()
        idx = range(i, i + n)
        project(
            ray_pos[idx],
            det_pos[idx],
            det_u[idx],
            det_v[idx],
            ball_pos,
            ball_radius,
            proj_data,
        )
        for p in proj_data.cpu().numpy():
            yield p

        del proj_data


def project(source_pos, det_pos, det_u, det_v, ball_pos, ball_radius,
            out_proj_data):
    cb.project_balls(
        source_pos, det_pos, det_u, det_v, ball_pos, ball_radius, out_proj_data
    )
    return out_proj_data


def save_spec(dir, ball_pos, ball_radius):
    num_balls = len(ball_radius)
    spec_path = os.path.join(dir, "cone_balls_spec.txt")
    spec = np.hstack((ball_pos, np.expand_dims(ball_radius, 1)))
    header = f"""
    This file was generated with cone_balls.

    There were {num_balls} balls projected onto a cone beam geometry.

    To load this file, execute:
    ---------------------------
    >>> ball_spec = np.loadtxt(file_path)
    >>> ball_pos = ball_spec[:, :3]
    >>> ball_radius = ball_spec[:, 3]
    """
    np.savetxt(spec_path, spec, header=header)


def load_spec(spec_path):
    ball_spec = np.loadtxt(spec_path)
    ball_pos = torch.tensor(ball_spec[:, :3], **torch_options)
    ball_radius = torch.tensor(ball_spec[:, 3], **torch_options)
    return ball_pos, ball_radius


def save_geometry(dir, pg):
    geom_path = os.path.join(dir, "astra_geometry.pickle")
    with open(geom_path, "wb") as handle:
        pickle.dump(pg, handle)


def load_geometry(geometry_path):
    with open(geometry_path, "rb") as handle:
        pg = pickle.load(handle)
    return pg


@click.group()
def main():
    pass


@main.command()
@click.option("--num_balls", default=100, help="Number of balls to generate.")
@click.option(
    "--ball_limit", default=200,
    help="The maximal distance from the origin of a ball"
)
@click.option("--num_angles", default=1500, help="Number of angles.")
@click.option("--det_row_count", default=700, help="Detector row count.")
@click.option("--det_col_count", default=700, help="Detector column count.")
@click.option("--pixel_size", default=1.0, help="The detector pixel size.")
@click.option("--SOD", default=700.0, help="The source object distance.")
@click.option("--SDD", default=700.0, help="The source detector distance.")
@click.option(
    "--interactive/--no-interactive",
    default=False,
    help="Show geometry and resulting projection images",
)
@click.option(
    "--ball_spec",
    default=None,
    type=click.Path(
        file_okay=True, dir_okay=False, resolve_path=True, allow_dash=False
    ),
)
@click.argument(
    "dir",
    type=click.Path(
        file_okay=False, dir_okay=True, resolve_path=True, allow_dash=False
    ),
)
def generate(
    num_balls,
    ball_limit,
    num_angles,
    det_row_count,
    det_col_count,
    pixel_size,
    sod,
    sdd,
    interactive,
    ball_spec,
    dir,
):
    """cone_balls generates ball phantoms for cone beam geometries
    """
    click.echo(f"Writing in {dir}!")

    pg = generate_projection_geometry(
        (pixel_size, pixel_size), (det_row_count, det_col_count),
        num_angles, sod, sdd
    )

    if ball_spec:
        logging.info(f"Not generating balls, using {ball_spec} file.")
        ball_pos, ball_radius = load_spec(ball_spec)
    else:
        logging.info(f"Generating {num_balls} balls.")
        ball_pos, ball_radius = generate_balls(num_balls, ball_limit)

    if interactive:
        pass  # Perhaps show geometry here in the future?

    proj_data = generate_projections(pg, ball_pos, ball_radius)

    if interactive:
        proj_data = np.array(list(proj_data))
        app = pq.mkQApp()
        pq.image(proj_data, axes={"t": 0, "x": 2, "y": 1})
        app.exec_()

    if not interactive:
        if os.path.exists(dir):
            warnings.warn(f"{dir} already exists. Overwriting files.")

        # Save ball positions:
        save_spec(dir, ball_pos, ball_radius)
        # Save geometry:
        save_geometry(dir, pg)
        # Save tiff stack:
        for i, p in tqdm(enumerate(proj_data)):
            filename = f"scan_{i:06d}.tif"
            path = os.path.join(dir, filename)
            tifffile.TiffWriter.save
            tifffile.imsave(path, p, metadata={"axes": "XY"})


@main.command()
@click.option("--num_balls", default=100, help="Number of balls to generate.")
@click.option(
    "--ball_limit", default=0.25,
    help="The maximal distance from the origin of a ball"
)
@click.option("--num_angles", default=100, help="Number of angles.")
@click.option("--det_row_count", default=1000, help="Detector row count.")
@click.option("--det_col_count", default=1000, help="Detector column count.")
@click.option("--pixel_size", default=1.2e-3, help="The detector pixel size.")
@click.option("--SOD", default=2.0, help="The source object distance.")
@click.option("--SDD", default=2.0, help="The source detector distance.")
@click.option(
    "--interactive/--no-interactive",
    default=False,
    help="Show geometry and resulting projection images",
)
@click.option(
    "--ball_spec",
    default=None,
    type=click.Path(
        file_okay=True, dir_okay=False, resolve_path=True, allow_dash=False
    ),
)
@click.argument(
    "dir",
    type=click.Path(
        file_okay=False, dir_okay=True, resolve_path=True, allow_dash=False
    ),
)
def foam(
    num_balls,
    ball_limit,
    num_angles,
    det_row_count,
    det_col_count,
    pixel_size,
    sod,
    sdd,
    interactive,
    ball_spec,
    dir,
):
    """cone_balls generates ball phantoms for cone beam geometries
    """
    click.echo(f"Writing in {dir}!")

    pg = generate_projection_geometry(
        (pixel_size, pixel_size), (det_row_count, det_col_count),
        num_angles, sod, sdd
    )

    if ball_spec:
        logging.info(f"Not generating balls, using {ball_spec} file.")
        ball_pos, ball_radius = load_spec(ball_spec)
    else:
        logging.info(f"Generating {num_balls} balls.")
        ball_pos, ball_radius = generate_balls(num_balls, ball_limit)

    foam_pos = torch.zeros(1, 3, **torch_options)
    foam_radius = torch.full((1,), 0.5, **torch_options)

    if interactive:
        pass  # Perhaps show geometry here in the future?

    ball_data = generate_projections(pg, ball_pos, ball_radius)
    foam_data = generate_projections(pg, foam_pos, foam_radius)

    if interactive:
        proj_data = np.array(list(foam_data)) - np.array(list(ball_data))
        app = pq.mkQApp()
        pq.image(proj_data, axes={"t": 0, "x": 2, "y": 1})
        app.exec_()

    if not interactive:
        if os.path.exists(dir):
            warnings.warn(f"{dir} already exists. Overwriting files.")

        # Save ball positions:
        save_spec(dir, ball_pos, ball_radius)
        # Save geometry:
        save_geometry(dir, pg)
        # Save tiff stack:
        for i, (ball, foam) in tqdm(enumerate(zip(ball_data, foam_data))):
            filename = f"scan_{i:06d}.tif"
            path = os.path.join(dir, filename)
            p = foam - ball
            tifffile.imsave(path, p, metadata={"axes": "XY"})


@main.command()
@click.option("--num_balls", default=100, help="Number of balls to generate.")
@click.option("--num_angles", default=1500, help="Number of angles.")
@click.option("--det_pix_count", default=700, help="Detector column count.")
@click.option(
    "--interactive/--no-interactive",
    default=False,
    help="Show geometry and resulting projection images",
)
def bench(
    num_balls,
    num_angles,
    det_pix_count,
    interactive,
):
    """Time the generation of cone_balls
    """

    ball_limit = 200
    pixel_size = 700 / det_pix_count
    sod = 700
    sdd = 700

    pg = generate_projection_geometry((pixel_size, pixel_size),
                                      (det_pix_count, det_pix_count),
                                      num_angles, sod, sdd)

    logging.info(f"Generating {num_balls} balls.")
    ball_pos, ball_radius = generate_balls(num_balls, ball_limit)

    if interactive:
        pass  # Perhaps show geometry here in the future?

    proj_data = generate_projections(pg, ball_pos, ball_radius)

    proj_data = np.array(list(proj_data))

    if interactive:
        app = pq.mkQApp()
        pq.image(proj_data, axes={"t": 0, "x": 2, "y": 1})
        app.exec_()


if __name__ == "__main__":
    main()
