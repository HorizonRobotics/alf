{ lib
, buildPythonPackage
, fetchPypi
, setuptools
, isPy39  
, isPy38
, isPy37 }:

assert (isPy39 || isPy38 || isPy37);

# TODO(breakds): Make this for other systems such as MacOSX and Windows.

buildPythonPackage rec {
  pname = "rectangle-packer";
  version = "2.0.1";
  format = "wheel";

  src = builtins.fetchurl (import ./wheel-urls.nix {
    inherit version isPy37 isPy38 isPy39; });

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
