{
  description = "Agent Learning Framework Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/22.05";

    utils.url = "github:numtide/flake-utils";
    utils.inputs.nixpkgs.follows = "nixpkgs";

    alf-devenv.url = "github:HorizonRobotics/alf-nix-devenv";
    alf-devenv.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, ... }@inputs: inputs.utils.lib.eachSystem [
    "x86_64-linux"
  ] (system: {
    devShell = inputs.alf-devenv.devShells."${system}".default;
  });
}
