#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

requirements = [
    "torchvision",
    "astra-toolbox",
    "Click",
    "pyqtgraph",
    "tifffile",
    "tqdm"
]

setup_requirements = []

test_requirements = []

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'bumpversion',
    'watchdog',
    'coverage',

    ]

setup(
    author="Allard Hendriksen",
    author_email='allard.hendriksen@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A phantom generation package for cone beam CT geometries.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cone_balls',
    name='cone_balls',
    entry_points='''
        [console_scripts]
        cone_balls=cone_balls:main
    ''',
    ext_modules=[
        CUDAExtension(
            name='cone_balls_cuda',
            sources=[
                'cone_balls/projector.cpp',
                'cone_balls/projector_cuda.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(include=['cone_balls']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={'dev': dev_requirements},
    url='https://github.com/ahendriksen/cone_balls',
    version='0.1.0',
    zip_safe=False,
)
