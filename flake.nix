{
  nixConfig = {
    # https://wiki.nixos.org/wiki/CUDA#NixOS
    extra-substituters = [
      "https://cache.nixos-cuda.org" # new
      "https://nix-community.cachix.org" # old, still used for some packages
    ];
    extra-trusted-public-keys = [
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs.nixpkgs-stable.url = "https://flakehub.com/f/NixOS/nixpkgs/0.2511.tar.gz";
  inputs.nixpkgs.follows = "nixpkgs-stable";

  inputs.flakelight = {
    # https://github.com/nix-community/flakelight
    url = "github:nix-community/flakelight";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    { flakelight, ... }@inputs:
    flakelight ./. (
      { lib, ... }:
      {
        inherit inputs;
        nixpkgs.config = {
          # cudaSupport used only for non-bin tensorflow/torch/torchvision which would require expensive rebuilds w/cuda, however, the builds are cached so it is ok
          # (cuda is included in the *-bin packages; see https://discourse.nixos.org/t/improving-nixos-data-science-infrastructure-ci-for-mkl-cuda/5074)
          cudaSupport = true;
          allowUnfree = true;
        };
        devShell =
          {
            python3,
            flower-framework,
            flower-datasets,
            ...
          }:
          {
            packages =
              let
                pythonPackages =
                  ps: with ps; [
                    # required by pytorch-federated-variational-autoencoder project
                    torchWithCuda # non-bin package build in cache.nixos-cuda.org
                    # DISABLED: as both {torch,torchvision}-bin require expensive rebuilds, e.g., of nccl; using cached non-bin packages instead
                    #torch-bin
                    #torchvision-bin
                    # required by tensorflow-federated-variational-autoencoder project
                    tensorflow-bin
                    # required by both
                    pandas
                    datasets
                    matplotlib
                    scikit-learn
                    (flower-datasets.override { python3Packages = ps; })
                    keras
                    wandb
                    # type-stubs
                    pandas-stubs
                  ];
              in
              [
                (flower-framework.override {
                  withExtraNonpropagatedPythonPackages = pythonPackages;
                })
                (python3.withPackages (
                  ps:
                  (pythonPackages ps)
                  ++ (with ps; [
                    # https://yannipapandreou.github.io/posts/jupyter-kernels-nix/index.html
                    #ipykernel jupyterlab # provides 'jupyter lab'
                    # https://nix-tutorial.gitlabpages.inria.fr/nix-tutorial/notebook.html
                    ipython
                    jupyter # provides 'jupyter notebook'
                  ])
                ))
              ];
            env = {
              TF_ENABLE_ONEDNN_OPTS = "0"; # disable oneDNN custom operations to prevent floating-point round-off errors from different computation orders
              RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO = "0"; # hide notification that Ray wont override accelerator visible devices env var if num_gpus=0 or num_gpus=None
            };
            shellHook = ''
              echo "Flower with Python packages is ready; cd to the project dir and run 'flwr run .' to launch."
              echo "Jupyter with Python kernel is ready; run 'jupyter notebook' to launch."
              echo "Run tensorflow-federated-variational-autoencoder with: cd ./tensorflow-federated-variational-autoencoder && flwr run --run-config 'dataset-path=\"../datasets/cic-aa.normal.tls/*.json\"' ."
              echo "To check available CPU/GPUs by Tensorflow, run: python3 -c 'from tensorflow.config import list_physical_devices; print(list_physical_devices())'"
              echo "To check available GPUs by PyTorch/CUDA, run: python3 -c 'import torch; print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])'"
            '';
          };
        flakelight.builtinFormatters = false; # do not format JSONs of dataset, etc.
        formatters = pkgs: {
          "*.nix" = lib.meta.getExe pkgs.nixfmt-rfc-style;
          "*.py" = "${lib.meta.getExe pkgs.ruff} format";
          "*.ipynb" = "${lib.meta.getExe pkgs.ruff} format";
          "*.toml" = "${lib.meta.getExe pkgs.toml-sort} --in-place";
        };
      }
    );
}
