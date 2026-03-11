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
  iterators,
  withExtraNonpropagatedPythonPackages ? (_: [ ]), # for extra non-propagated dependencies of apps run inside the Flower by "flwr run"
  # BUG: Flower w/new protobuf is not compatible with Tensorflow; it works ok in both runtime and data processing, however, Tensorflow reports warning
  # protobuf(6) throws "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'" because Google deprecated MessageFactory some time back
  # WORKAROUND: extracting "protobuf4" and some of internally used deps with protobuf from propagatedBuildInputs into buildInputs does help, however, multiple protobuf versions may result into "UnicodeDecodeError" on: python3 -c "import tensorflow"
  # Tensorflow issue: https://github.com/tensorflow/tensorflow/issues/94030
  # https://github.com/microsoft/PhiCookBook/issues/286#issuecomment-2798922731
  withLegacyProtobuf ? false, # FALSE to prevent multiple protobuf version that would result into "UnicodeDecodeError" on: python3 -c "import tensorflow"
}:

let

  nvFetcherSrc = sources.default;

  inherit (nvFetcherSrc) src version;

  inherit (outputs'.packages) iterators;

in
python3Packages.buildPythonPackage {

  inherit src version;

  pname = "flower-framework";

  format = "pyproject";

  sourceRoot = "./source/framework";

  patches = [
    ./0001-loosen-version-requirement-v1.25.0.patch
  ];

  nativeBuildInputs = [
    # build-system
    python3Packages.poetry-core
  ];

  buildInputs =
    with python3Packages;
    [
      # https://github.com/adap/flower/blob/main/framework/pyproject.toml
      # tool.poetry.dependencies
      grpcio # uses protobuf6, however, it is needed only internally; building with protobuf4 is expensive
    ]
    ++ (withExtraNonpropagatedPythonPackages python3Packages);

  propagatedBuildInputs = with python3Packages; [
    # https://github.com/adap/flower/blob/main/framework/pyproject.toml
    # tool.poetry.dependencies
    numpy
    (if withLegacyProtobuf then protobuf4 else protobuf)
    cryptography
    pycryptodome
    iterators # not in nixpkgs
    typer
    tomli
    tomli-w
    pathspec
    rich
    pyyaml
    requests
    click # only for >=v1.19.0
    grpcio-health-checking # only for >=v1.25.0
    # Optional dependencies (Simulation Engine)
    (if withLegacyProtobuf then ray.override { protobuf = protobuf4; } else ray)
  ];

  meta = with lib; {
    longDescription = "A unified approach to federated learning, analytics, and evaluation. Federate any workload, any ML framework, and any programming language.";
    description = "A Friendly Federated Learning Framework.";
    homepage = "https://flower.dev/";
    inherit (python3Packages.python.meta) platforms;
    license = licenses.asl20;
  };
}
