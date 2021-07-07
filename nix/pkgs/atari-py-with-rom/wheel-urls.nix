# The sha256 in this file can be fetched by calling
#
# nix-prefetch-url <URL>

{ version, isPy37, isPy38, isPy39 }:

let urls = {
      "0.2.9" = {
        py37 = {
          url = https://files.pythonhosted.org/packages/7a/ad/bf0b26d4aa571e393619bd4d77e6ccb45f39a23d87f9a67080e02fa7b831/atari_py-0.2.9-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
          sha256 = "15zb67kdcxifrch9baqdpm7dbx53n96p2r8hx5lbdszypmfff6al";
        };

        py38 = {
          url = https://files.pythonhosted.org/packages/02/6c/4195016867435a7b7b0b6b89be70dbd1f67a8d882918bfe122974a8d98cd/atari_py-0.2.9-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
          sha256 = "0v30hwj280c0dcy5b9fawknnd4zrqi49zkfr8pd4b03hzy8xwm2d";
        };

        py39 = {
          url = https://files.pythonhosted.org/packages/67/88/f1db6f411de4285281f56c455042a67c77c2f0c5c4cf6a1c5ea41e0fabee/atari_py-0.2.9-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
          sha256 = "0yqlm7s967sb7gsc123k663c1k9b1v5ihy99f5glcc1kb9z3nw59";
        };
      };
    };
in (if isPy37 then urls."${version}".py37
    else if isPy38 then urls."${version}".py38
    else urls."${version}".py39)
