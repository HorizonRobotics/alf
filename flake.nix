{
  description = "Agent Learning Framework Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

    utils.url = "github:numtide/flake-utils";

    ml-pkgs.url = "github:nixvital/ml-pkgs";
    ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
    ml-pkgs.inputs.utils.follows = "utils";
  };

  outputs = { self, nixpkgs, ml-pkgs, ... }@inputs: {
    overlays.extra = final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          cnest = python-final.callPackage ./nix/pkgs/cnest {};
          rectangle-packer = python-final.callPackage ./nix/pkgs/rectangle-packer {};
          pybox2d = python-final.callPackage ./nix/pkgs/pybox2d {};
          gin-config = python-final.callPackage ./nix/pkgs/gin-config {};
          pre-commit = python-final.callPackage ./nix/pkgs/pre-commit {};
        })
      ];
    };
    overlays.default = nixpkgs.lib.composeManyExtensions [
      inputs.ml-pkgs.overlays.torch-family
      inputs.ml-pkgs.overlays.simulators
      self.overlays.extra
      (final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (python-final: python-prev: {
            alf = python-final.callPackage ./default.nix {};
          })
        ];
      })
    ];
  } // inputs.utils.lib.eachSystem [
    "x86_64-linux"
  ] (system: {
    packages = let pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaCapabilities = [ "7.5" "8.6" ];
        cudaForwardCompat = true;
      };
      overlays = [
        self.overlays.default
      ];
    }; in {
      default = pkgs.python3Packages.alf;
    };
  });
}
