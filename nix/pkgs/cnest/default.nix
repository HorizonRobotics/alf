{ lib
, buildPythonPackage
, fetchFromGitHub
, pybind11
, setuptools
}:

buildPythonPackage rec {
  pname = "cnest";
  version = "1.1.1";

  src = fetchFromGitHub {
    owner = "HorizonRobotics";
    repo = pname;
    rev = "b7a62849ac4531225229cff3a5d5f8fc654bda3f";
    hash = "sha256-TWmB3BbLjlYWPd+qMg26C3ladHDleulIsfhx31xzSn8=";
  };

  buildInputs = [
    pybind11
    setuptools
  ];

  doCheck = false;

  meta = with lib; {
    homepage = "https://github.com/HorizonRobotics/alf/tree/pytorch/alf/nest";
    description = "C++ implementation of several key nest functions that are preformance critical";
    license = licenses.mit;
    maintainers = with maintainers; [ breakds ];
  };
}
