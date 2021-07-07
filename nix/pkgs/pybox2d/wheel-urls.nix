# The sha256 in this file can be fetched by calling
#
# nix-prefetch-url <URL>

{ version, isPy37, isPy38 }:

let urls = {
      "2.3.10" = {
        py37 = {
          url = https://files.pythonhosted.org/packages/22/1b/ce95bb5d1807d4d85af8d0c90050add1a77124459f8097791f0c39136d53/Box2D-2.3.10-cp37-cp37m-manylinux1_x86_64.whl;
          sha256 = "0rhj7yxqmc71x8anrj0wxbahq84h2g9b5z30idhgq3qka53j4zl2";
        };

        py38 = {
          url = https://files.pythonhosted.org/packages/72/42/6a8e18a93f75c84fd065cc9b57a4117219fa3c5a002a80cab8f339883ec8/Box2D-2.3.10-cp38-cp38-manylinux1_x86_64.whl;
          sha256 = "1agyk2axvlq8zfq9wl676c0asx6c4wr511l0x7sxgw8xfnn7aac0";
        };
      };
    };
in (if isPy37 then urls."${version}".py37
    else if isPy38 then urls."${version}".py38
    else urls."${version}".py39)
