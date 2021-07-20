# This is a development environment for agent learning framework.

{ mkShell, python3, python-language-server, clang-tools }:

let pythonForAlf = python3.withPackages (pyPkgs: with pyPkgs; [
      # For both Dev and Deploy
      pytorchWithCuda11
      torchvisionWithCuda11
      numpy pandas absl-py
      gym
      # TODO(breakds): Require pyglet 1.3.2, because higher version
      # breaks classic control rendering. Or fix the classic control
      # rendering.
      pyglet
      opencv4
      pathos
      pillow
      psutil
      pybullet
      sphinx
      gin-config
      cnest
      fasteners
      rectangle-packer
      pybox2d
      atari-py-with-rom
      # TODO(breakds): Package torchtext and enable it.
      # torchtext (0.9.1)
      
      # Dev only packages
      jupyterlab ipywidgets ipydatawidgets
      matplotlib tqdm
      sphinx_rtd_theme
      yapf
      pre-commit
      pylint
      pudb
    ]);

    pythonIcon = "f3e2";

in mkShell rec {
  name = "ALF";

  packages = [
    pythonForAlf
    python-language-server  # From Microsoft, not Palantir
  ];

  # This is to have a leading python icon to remind the user we are in
  # the Alf python dev environment.
  shellHook = ''
    export PS1="$(echo -e '\u${pythonIcon}') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} (${name}) \\$ \[$(tput sgr0)\]"
    export PYTHONPATH="$(pwd):$PYTHONPATH"
  '';
}
