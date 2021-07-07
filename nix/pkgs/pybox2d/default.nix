{ lib
, buildPythonPackage
, autoPatchelfHook
, isPy37
, isPy38
, stdenv
}:

assert (isPy37 || isPy38);

buildPythonPackage rec {
  pname = "Box2D";
  version = "2.3.10";
  format = "wheel";

  src = builtins.fetchurl (import ./wheel-urls.nix {
    inherit version isPy37 isPy38; });

  buildInputs = [
    stdenv.cc.cc.lib
  ];

  nativeBuildInputs = [
    autoPatchelfHook
  ];


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
