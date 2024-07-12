{ lib
, buildPythonPackage
, fetchFromGitHub
, swig2
}:

buildPythonPackage rec {
  pname = "Box2D";
  version = "2.3.10";

  src = fetchFromGitHub {
    owner = "pybox2d";
    repo = "pybox2d";
    rev = "2.3.10";
    sha256 = "sha256-yjLFvsg8GQLxjN1vtZM9zl+kAmD4+eS/vzRkpj0SCjY=";
  };

  nativeBuildInputs = [ swig2 ];

  # tests not included with pypi release
  doCheck = false;

  meta = with lib; {
    homepage = "https://github.com/pybox2d/pybox2d";
    description = ''
      A 2D game physics library for Python under
      the very liberal zlib license
    '';
    license = licenses.zlib;
    maintainers = with maintainers; [ breakds ];
  };
}
