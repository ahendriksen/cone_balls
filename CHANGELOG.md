# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Python 3.7 Conda package
- Parallel beam projections
### Fixed
- Bug where output directory would not be created
### Changed
- Upgraded to pytorch 1.1.0
### Removed
- Support for cudatoolkit=9.0

## [0.2.2] - 2019-05-09
### Added
- Add --Z option to foam command to move source and detector up.
- Add example code to README.md to load cone_balls data.
### Fixed
- The --interactive option now correctly displays projections upside
  up instead of upside down.

## [0.2.1] - 2018-12-21
### Added
- foam command to generate foam bubble artifacts.

## [0.2.0] - 2018-12-21
### Added
- Add a foam command to generate foam bubble artifacts.
- Add a benchmark command
### fixed
- Improve performance by using fewer CUDA registers


## 0.1.0 - 2018-11-14
### Added
- Initial release.

[Unreleased]: https://www.github.com/ahendriksen/cone_balls/compare/v0.2.2...HEAD
[0.2.2]: https://www.github.com/ahendriksen/cone_balls/compare/v0.2.1...v0.2.2
[0.2.1]: https://www.github.com/ahendriksen/cone_balls/compare/v0.2.0...v0.2.1
[0.2.0]: https://www.github.com/ahendriksen/cone_balls/compare/v0.1.0...v0.2.0
