{ lib
, buildPythonPackage
, fetchPypi
, setuptools
, python }:


let pyVerNoDot = builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;
    
in buildPythonPackage rec {
  pname = "rectangle-packer";
  version = "2.0.2";
  format = "wheel";

  src = builtins.fetchurl (import ./wheel-urls.nix {
    inherit version pyVerNoDot; });

  propagatedBuildInputs = [ setuptools ];

  meta = with lib; {
    description = ''
      Given a set of rectangles with fixed orientations, find a bounding box of 
      minimum area that contains them all with no overlap.
    '';
    homepage = "https://github.com/Penlect/rectangle-packer";
    license = licenses.mit;
    maintainers = with maintainers; [ breakds ];
  };
}
