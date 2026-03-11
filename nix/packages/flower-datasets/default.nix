{
  outputs',
  lib,
  python3Packages,
  sources ? import ./_sources/generated.nix {
    inherit
      fetchurl
      fetchgit
      fetchFromGitHub
      dockerTools
      ;
  },
  fetchurl,
  fetchgit,
  fetchFromGitHub,
  dockerTools,
}:

let

  nvFetcherSrc = sources.default;

  inherit (nvFetcherSrc) src version;

  inherit (outputs'.packages) iterators;

in
python3Packages.buildPythonPackage {

  inherit src version;

  pname = "flower-datasets";

  format = "pyproject";

  sourceRoot = "./source/datasets";

  patches = [
    ./0001-loosen-version-requirement-v1.19.0.patch
  ];

  postPatch = ''
    # >=v1.19.0 requires poetry-core>=2.1.3 which is not yet updated in nixpkgs
    substituteInPlace pyproject.toml --replace-fail "poetry-core>=2.1.3" "poetry-core"
  '';

  nativeBuildInputs = [
    # build-system
    python3Packages.poetry-core
  ];

  propagatedBuildInputs = with python3Packages; [
    # https://github.com/adap/flower/blob/main/datasets/pyproject.toml
    # tool.poetry.dependencies
    numpy
    datasets
    tqdm
    matplotlib
    seaborn
  ];

  meta = with lib; {
    longDescription = "A library to quickly and easily create datasets for federated learning, federated evaluation, and federated analytics.";
    description = "A Friendly Federated Learning Datasets Library.";
    homepage = "https://flower.dev/";
    inherit (python3Packages.python.meta) platforms;
    license = licenses.asl20;
  };
}
