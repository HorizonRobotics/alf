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
          overlays = [
            # Use this overlay to provide customized python packages
            # for development environment.
            (import ./nix/overlays/dev.nix)
          ];
        };
    in {
      # Instantiate the development environment with CUDA 11.2
      devShell = pkgs.callPackage ./nix/pkgs/alf-dev-shell {};
    });
}
