{ lib, buildPythonPackage, pybind11 }:

buildPythonPackage rec {
  pname = "cnest";
  version = "1.0-trunk";

  src = ../../../alf/nest/cnest;

  buildInputs = [
    pybind11
  ];

  doCheck = false;

  meta = with lib; {
    homepage = "https://github.com/HorizonRobotics/alf/tree/pytorch/alf/nest";
    description = "C++ implementation of several key nest functions that are preformance critical";
    license = licenses.asl20;
    maintainers = with maintainers; [ breakds ];
  };
}
