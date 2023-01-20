{
  description = "Agent Learning Framework Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";

    utils.url = "github:numtide/flake-utils";

    alf-devenv.url = "github:HorizonRobotics/alf-nix-devenv";
    alf-devenv.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, ... }@inputs: {
    overlays.default = final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          alf = python-final.callPackage ./default.nix {};
          alf-cpu = python-final.callPackage ./default.nix { useCuda = false; };
        })
      ];
    };
  } // inputs.utils.lib.eachSystem [
    "x86_64-linux"
  ] (system: {
    devShells.default = inputs.alf-devenv.devShells."${system}".default;
    packages = let pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      overlays = [
        inputs.alf-devenv.overlays.default        
        self.overlays.default
      ];
    }; in {
      default = pkgs.python3Packages.alf;
      cpu = pkgs.python3Packages.alf-cpu;
    };
  });
}
