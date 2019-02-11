# Cone Balls

A phantom generation package for cone beam CT geometries.

This generator analytically computes a circular cone beam projection
of a configurable number of (potentially overlapping) solid spheres of
constant density and varying random radius.

It randomly places the centers of the spheres in a cube of
configurable size. Where the spheres overlap, their linear attenuation
coefficients are summed.

The randomly generated ball positions and radii are stored for later
re-use, as well as the used cone beam geometry (as an ASTRA projection
geometry).

Cone Balls is GPU accelerated and requires CUDA.


* Free software: GNU General Public License v3
* Documentation: [https://ahendriksen.github.io/cone_balls]


## Readiness

The author of this package is in the process of setting up this
package for optimal usability. The following has already been completed:

- [x] Documentation
    - Documentation has been generated using `make docs`, committed,
        and pushed to GitHub.
	- GitHub pages have been setup in the project
      [Settings](/settings) with "master branch /docs folder".
- [ ] An initial release
	- In `CHANGELOG.md`, a release date has been added to v0.1.0 (change the YYYY-MM-DD).
	- The release has been marked a release on GitHub.
	- For more info, see the [Software Release Guide](https://cicwi.github.io/software-guidelines/software-release-guide).
- [x] A conda package
	- Required packages have been added to `setup.py`, for instance,
	  ```
	  requirements = [ ]
	  ```
	  Has been replaced by
	  ```
	  requirements = [
	      'sacred>=0.7.2'
      ]
      ```
	- All "conda channels" that are required for building and
      installing the package have been added to the
      `Makefile`. Specifically, replace
	  ```
      conda_package: install_dev
      	conda build conda/
      ```
	  by
	  ```
      conda_package: install_dev
      	conda build conda/ -c some-channel -c some-other-channel
      ```
    - Conda packages have been built successfully with `make conda_package`.
	- These conda packages have been uploaded to [Anaconda](https://anaconda.org).
	- The installation instructions (below) have been updated.

## Getting Started

It takes a few steps to setup Cone Balls on your
machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) for
Python 3.

### Installing with conda

Simply install with:
```
conda install -c aahendriksen -c pytorch -c astra-toolbox/label/dev -c conda-forge cone_balls
```

### Installing from source

To install Cone Balls, simply clone this GitHub
project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/ahendriksen/cone_balls.git
cd cone_balls
conda install -c astra-toolbox/label/dev -c pytorch -c conda-forge cuda92 astra-toolbox pytorch torchvision tifffile
pip install -e .
```

### Usage

Cone balls is a command-line utility. It creates a directory with tiff files.

``` bash
$ cone_balls --help
Usage: cone_balls [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  bench     Time the generation of cone_balls
  generate  cone_balls generates ball phantoms for cone beam geometries

$ cone_balls generate --help
Usage: cone_balls generate [OPTIONS] DIR

  generate generates cone-beam projections of ball phantoms

  By default - 100 balls are randomly generated - 1500 projections are
  computed on a 700 x 700 detector with pixel size 1.0 x 1.0 - The source-
  object distance and the source-detector distance are 700.0, meaning   that
  the detector is centered on the origin and rotates through the object.

Options:
  --num_balls INTEGER             Number of balls to generate.
  --ball_limit INTEGER            The maximal distance from the origin of a
                                  ball
  --num_angles INTEGER            Number of angles.
  --det_row_count INTEGER         Detector row count.
  --det_col_count INTEGER         Detector column count.
  --pixel_size FLOAT              The detector pixel size.
  --SOD FLOAT                     The source object distance.
  --SDD FLOAT                     The source detector distance.
  --interactive / --no-interactive
                                  Show geometry and resulting projection
                                  images
  --ball_spec FILE
  --help                          Show this message and exit.

$ cone_balls foam --help
Usage: cone_balls foam [OPTIONS] DIR

  foam generates cone-beam projections of a foam ball phantom

  The foam ball has a radius of 0.5 and is centered on the origin. Bubbles
  can be removed from this foam phantom. The location and size of these
  bubbles can either be supplied using the --ball_spec option, or randomly
  generated.

Options:
  --num_balls INTEGER             Number of balls to generate.
  --ball_limit FLOAT              The maximal distance from the origin of a
                                  ball
  --num_angles INTEGER            Number of angles.
  --det_row_count INTEGER         Detector row count.
  --det_col_count INTEGER         Detector column count.
  --pixel_size FLOAT              The detector pixel size.
  --SOD FLOAT                     The source object distance.
  --SDD FLOAT                     The source detector distance.
  --Z FLOAT                       The Z-offset of source and detector.
  --interactive / --no-interactive
                                  Show geometry and resulting projection
                                  images
  --ball_spec FILE
  --help                          Show this message and exit.

$ cone_balls bench --help
Usage: cone_balls bench [OPTIONS]

  Time the generation of cone_balls

Options:
  --num_balls INTEGER             Number of balls to generate.
  --num_angles INTEGER            Number of angles.
  --det_pix_count INTEGER         Detector column count.
  --interactive / --no-interactive
                                  Show geometry and resulting projection
                                  images
  --help                          Show this message and exit.

```


### An example run

First generate projection data using cone_balls:
```
mkdir -p example
cone_balls generate --num_balls 10 --num_angles 50 example
```

Start a python console and run:
``` python
import astra
import astra.experimental
import numpy as np
from pathlib import Path
import tifffile
import pickle

# Load data
tiffs = [tifffile.imread(str(p)) for p in sorted(Path("example/").glob("*.tif"))]
# Put projection data in [Y, Angle, X] order
proj_data = np.array(tiffs).swapaxes(0, 1)

# Load projection geometry
pg =  pickle.load(open("example/astra_geometry.pickle", "rb"))
# Create astra projection data
p_id = astra.data3d.create('-proj3d', pg, proj_data)

vg = {'option': {'WindowMinX': -350, 'WindowMaxX': 350,
                 'WindowMinY': -350, 'WindowMaxY': 350,
                 'WindowMinZ': -350, 'WindowMaxZ': 350},
      'GridRowCount': 250,
      'GridColCount': 250,
      'GridSliceCount': 250
      }
v_id = astra.data3d.create('-vol', vg)

projector = astra.create_projector("cuda3d", pg, vg)
astra.experimental.accumulate_FDK(projector, v_id, p_id)

# Display volume data
import pyqtgraph as pq
app = pq.mkQApp()
pq.image(astra.data3d.get_shared(v_id))
app.exec_()
```


## Authors and contributors

* **Allard Hendriksen ** - *Initial work*

See also the list of [contributors](https://github.com/ahendriksen/cone_balls/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements
* To Willem Jan Palenstijn for useful advice and discussion.
