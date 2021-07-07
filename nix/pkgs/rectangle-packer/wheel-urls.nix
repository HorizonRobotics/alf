# The sha256 in this file can be fetched by calling
#
# nix-prefetch-url <URL>

{ version, isPy37, isPy38, isPy39 }:

let urls = {
      "2.0.1" = {
        py37 = {
          url = https://files.pythonhosted.org/packages/62/24/9ddaf1d3e0e88d9866a0d67ad5d3c9d3f82ea5f819435d76f6654e1fddf2/rectangle_packer-2.0.1-cp37-cp37m-manylinux2010_x86_64.whl;
          sha256 = "0gfcmwr7k1ifrmk7mwzfzyp8hh163mrjik572xn1d4j53l78qq5h";
        };

        py38 = {
          url = https://files.pythonhosted.org/packages/a5/83/13f95641e7920c471aff5db609e8ccff1f4204783aff63ff4fd51229389e/rectangle_packer-2.0.1-cp38-cp38-manylinux2010_x86_64.whl;
          sha256 = "00z2dnjv5pl44szv8plwlrillc3l7xajv6ncdf5sqxkb0g0r3kc6";
        };

        py39 = {
          url = https://files.pythonhosted.org/packages/c6/f3/2ca57636419c42b9a698a6378ed99a61bcff863db53a1ec40f0edd996099/rectangle_packer-2.0.1-cp39-cp39-manylinux2010_x86_64.whl;
          sha256 = "1kxy7kqs6j9p19aklx57zjsbmnrvqngs6zdi2s8c4qvshm3zzayk";
        };
      };
    };
in (if isPy37 then urls."${version}".py37
    else if isPy38 then urls."${version}".py38
    else urls."${version}".py39)
