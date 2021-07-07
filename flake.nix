{
  description = "Agent Learning Framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/21.05";

    utils.url = "github:numtide/flake-utils";
    utils.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, ... }@inputs: inputs.utils.lib.eachSystem [
    "x86_64-linux" "i686-linux" "aarch64-linux" "x86_64-darwin"
  ] (system:
    let pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
    in {
      # Instantiate the development environment with CUDA 11.2
      devShell = pkgs.callPackage ./nix/pkgs/alf-dev-shell {
        cudatoolkit = pkgs.cudatoolkit_11_2;
        nccl = pkgs.nccl_cudatoolkit_11;
        cudnn = pkgs.cudnn_cudatoolkit_11_2;
      };
    });
}
