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
from pathlib import Path

torch_options = {"dtype": torch.float32, "device": torch.device(type="cuda")}
BATCH = 500


def generate_cone_pg(det_spacing, det_shape, num_angles, SOD, SDD):
    angles = np.linspace(0, 2 * np.pi, num_angles, False)
    pg = astra.create_proj_geom(
        "cone", *det_spacing, *det_shape, angles, SOD, SDD - SOD
    )
    return astra.geom_2vec(pg)


def generate_parallel_pg(det_spacing, det_shape, num_angles):
    angles = np.linspace(0, 2 * np.pi, num_angles, False)
    pg = astra.create_proj_geom(
        "parallel3d", *det_spacing, *det_shape, angles
    )
    return astra.geom_2vec(pg)


def move_source(pg, offset):
    ret = pg.copy()
    ret["Vectors"] = np.copy(ret["Vectors"])
    src_z = ret["Vectors"][:, 2]

    src_z += offset

    return ret


def move_detector(pg, offset):
    ret = pg.copy()
    ret["Vectors"] = np.copy(ret["Vectors"])
    det_z = ret["Vectors"][:, 5]

    det_z += offset
    return ret


def generate_balls(num_balls, pos_limit):
    ball_pos = (0.5 - torch.rand(num_balls, 3, **torch_options)) * 2.0 * pos_limit
    ball_radius = torch.rand(num_balls, **torch_options) * pos_limit / 10
    return (ball_pos, ball_radius)


def generate_projections(pg, ball_pos, ball_radius, cone=True):
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

    generated_projections = torch.empty(num_angles, *det_shape, dtype=torch.float32).pin_memory()
    for i in range(0, num_angles, BATCH):
        # Generate BATCH number of projection images (or n, if there
        # are fewer remaining angles)
        n = min(BATCH, num_angles - i)
        proj_data = torch.zeros(n, *det_shape, **torch_options)
        proj_data.zero_()
        idx = slice(i, i + n)
        project(
            ray_pos[idx],
            det_pos[idx],
            det_u[idx],
            det_v[idx],
            ball_pos,
            ball_radius,
            proj_data,
            cone
        )
        generated_projections[idx] = proj_data.cpu()
        del proj_data

    return generated_projections.numpy()


def project(source_pos, det_pos, det_u, det_v, ball_pos, ball_radius,
            out_proj_data, cone=True):
    cb.project_balls(
        source_pos, det_pos, det_u, det_v, ball_pos, ball_radius, out_proj_data, cone
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


def display_projections(proj_data):
    app = pq.mkQApp()
    # Flip the image vertically. Pyqtgraph displays the first row of
    # the image on top. Astra expects the first row of the image to be
    # on the bottom of the detector (in the negative direction of the
    # v vector).
    proj_data = proj_data[:, ::-1, :]
    pq.image(proj_data, axes={"t": 0, "x": 2, "y": 1},
             title="Cone balls: projection data")
    app.exec_()


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
    "--cone/--parallel",
    default=True,
    help="Cone-beam geometry",
)
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
    cone,
    interactive,
    ball_spec,
    dir,
):
    """generate generates cone-beam projections of ball phantoms

    By default
    - 100 balls are randomly generated
    - 1500 projections are computed on a 700 x 700 detector with pixel size 1.0 x 1.0
    - The source-object distance and the source-detector distance are 700.0, meaning
      that the detector is centered on the origin and rotates through the object.
    """
    dir = Path(dir).expanduser().resolve()
    click.echo(f"Writing in {dir}!")
    dir.mkdir(exist_ok=True)

    if cone:
        pg = generate_cone_pg(
            (pixel_size, pixel_size),
            (det_row_count, det_col_count),
            num_angles,
            sod,
            sdd
        )
    else:
        pg = generate_parallel_pg(
            (pixel_size, pixel_size),
            (det_row_count, det_col_count),
            num_angles
        )

    if ball_spec:
        logging.info(f"Not generating balls, using {ball_spec} file.")
        ball_pos, ball_radius = load_spec(ball_spec)
    else:
        logging.info(f"Generating {num_balls} balls.")
        ball_pos, ball_radius = generate_balls(num_balls, ball_limit)

    if interactive:
        pass  # Perhaps show geometry here in the future?

    proj_data = generate_projections(pg, ball_pos, ball_radius, cone=cone)

    if interactive:
        proj_data = np.array(proj_data, copy=False)
        display_projections(proj_data)

    if not interactive:
        # Save ball positions:
        ball_pos, ball_radius = ball_pos.cpu().numpy(), ball_radius.cpu().numpy()
        save_spec(dir, ball_pos, ball_radius)
        # Save geometry:
        save_geometry(dir, pg)
        # Save tiff stack:
        for i, p in tqdm(enumerate(proj_data)):
            filename = f"scan_{i:06d}.tif"
            path = os.path.join(dir, filename)
            tifffile.imsave(path, p)


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
@click.option("--Z", default=0.0, help="The Z-offset of source and detector.")
@click.option(
    "--cone/--parallel",
    default=True,
    help="Cone-beam geometry",
)
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
    z,
    cone,
    interactive,
    ball_spec,
    dir,
):
    """foam generates cone-beam projections of a foam ball phantom

    The foam ball has a radius of 0.5 and is centered on the origin.
    Bubbles can be removed from this foam phantom.
    The location and size of these bubbles can either be supplied
    using the --ball_spec option, or randomly generated.

    """
    dir = Path(dir).expanduser().resolve()
    click.echo(f"Writing in {dir}!")
    dir.mkdir(exist_ok=True)

    if cone:
        pg = generate_cone_pg(
            (pixel_size, pixel_size),
            (det_row_count, det_col_count),
            num_angles,
            sod,
            sdd
        )
        pg = move_source(pg, z)
        pg = move_detector(pg, z)
    else:
        pg = generate_parallel_pg(
            (pixel_size, pixel_size),
            (det_row_count, det_col_count),
            num_angles
        )
        pg = move_detector(pg, z)

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

    ball_data = generate_projections(pg, ball_pos, ball_radius, cone=cone)
    foam_data = generate_projections(pg, foam_pos, foam_radius, cone=cone)

    if interactive:
        proj_data = foam_data - ball_data
        display_projections(proj_data)

    if not interactive:
        if os.path.exists(dir):
            warnings.warn(f"{dir} already exists. Overwriting files.")

        # Save ball positions:
        ball_pos, ball_radius = ball_pos.cpu().numpy(), ball_radius.cpu().numpy()
        save_spec(dir, ball_pos, ball_radius)
        # Save geometry:
        save_geometry(dir, pg)
        # Save tiff stack:
        for i, (ball, foam) in tqdm(enumerate(zip(ball_data, foam_data))):
            filename = f"scan_{i:06d}.tif"
            path = os.path.join(dir, filename)
            p = foam - ball
            tifffile.imsave(path, p)


@main.command()
@click.option("--num_balls", default=100, help="Number of balls to generate.")
@click.option("--num_angles", default=1500, help="Number of angles.")
@click.option("--det_pix_count", default=700, help="Detector column count.")
@click.option(
    "--cone/--parallel",
    default=True,
    help="Cone-beam geometry",
)
@click.option(
    "--interactive/--no-interactive",
    default=False,
    help="Show geometry and resulting projection images",
)
def bench(
    num_balls,
    num_angles,
    det_pix_count,
    cone,
    interactive,
):
    """Time the generation of cone_balls
    """

    ball_limit = 200
    pixel_size = 700 / det_pix_count
    sod = 700
    sdd = 700

    if cone:
        pg = generate_cone_pg(
            (pixel_size, pixel_size),
            (det_pix_count, det_pix_count),
            num_angles,
            sod,
            sdd
        )
    else:
        pg = generate_parallel_pg(
            (pixel_size, pixel_size),
            (det_pix_count, det_pix_count),
            num_angles
        )

    logging.info(f"Generating {num_balls} balls.")
    ball_pos, ball_radius = generate_balls(num_balls, ball_limit)

    if interactive:
        pass  # Perhaps show geometry here in the future?

    proj_data = generate_projections(pg, ball_pos, ball_radius, cone=cone)

    if interactive:
        display_projections(proj_data)


if __name__ == "__main__":
    main()
