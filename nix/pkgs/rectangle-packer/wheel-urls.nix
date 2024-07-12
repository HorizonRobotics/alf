# The sha256 in this file can be fetched by calling
#
# nix-prefetch-url <URL>

{ version, pyVerNoDot }:

let urls = {
      "2.0.2" = {
        "311" = {
          url = https://files.pythonhosted.org/packages/e6/f4/cb4c11e212679a960384cba619180bf3058fc4798617cf87663dbf4a1c7b/rectangle_packer-2.0.2-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
          sha256 = "0xcpym1fywaj46sn4r9xrfc5bx63gkjd0vry2fm596sqfraz8dnh";
        };
        "310" = {
          url = https://files.pythonhosted.org/packages/22/93/a76935e161f862ae17bc0ea6d1a73c90476483bcd61f73464854abc6391e/rectangle_packer-2.0.2-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
          sha256 = "0b8rnh1jg00pf6dyvm1v7rgaz6hkx0r96xbbn7cmnmam2b7rav1q";
        };
      };
    };
in urls."${version}"."${pyVerNoDot}"
