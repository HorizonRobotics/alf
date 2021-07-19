{ lib
, buildPythonPackage
, fetchFromGitHub
, six
}:

buildPythonPackage rec {
  pname = "gin-config";
  version = "0.4.0";

  src = fetchFromGitHub {
    owner = "HorizonRobotics";
    repo = pname;
    rev = "ef217ba9e5cce69e2dc723507cd2b3563de9c5f6";
    sha256 = "sha256-lQV+BKce1kfKmBInzjDHIXlExg8kU7r2PsqvJ1K7gYg=";
  };

  propagatedBuildInputs = [ six ];

  # PyPI archive does not ship with tests
  doCheck= false;

  meta = with lib; {
    homepage = "https://github.com/google/gin-config";
    description = "Gin provides a lightweight configuration framework for Python, based on dependency injection.";
    license = licenses.asl20;
    maintainers = with maintainers; [ breakds ];
  };
}
