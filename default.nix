{ lib
, buildPythonPackage
, fetchFromGitHub
, pythonRelaxDepsHook
, pybind11
, boost
, torchWithCuda
, torchvision
, numpy
, einops
, wandb
, tensorboard
, absl-py
, pyglet
, opencv4
, pathos
, pillow
, psutil
, gin-config
, cnest
, fasteners
, rectangle-packer
, pybox2d
, matplotlib
, tqdm
, gym
, pybullet
, atari-py-with-rom
, procgen
, highway-env
, metadrive-simulator
, threadpoolctl
}:

buildPythonPackage rec {
  pname = "alf";
  version = "0.0.6";

  src = ./.;

  nativeBuildInputs = [
    pythonRelaxDepsHook
  ];

  buildInputs = [
    pybind11
    boost
  ];

  pythonRelaxDeps = [
    "matplotlib"
    "torchvision"
    "h5py"
    "pillow"
    "pyglet"
    "pybullet"
    "torch"
    "pathos"
    "atari-py"
    "rectangle-packer"
    "procgen"
    "tensorboard"
    "gym"
  ];

  propagatedBuildInputs = [
    # Machine Learning
    torchWithCuda
    torchvision
    numpy
    einops
    wandb
    tensorboard

    # Utils
    absl-py
    pyglet
    opencv4
    pathos
    pillow
    psutil
    gin-config
    cnest
    fasteners
    rectangle-packer
    pybox2d
    matplotlib
    tqdm
    threadpoolctl

    # Simulators
    gym
    pybullet
    atari-py-with-rom
    procgen
    highway-env
    metadrive-simulator
  ];

  doCheck = false;

  pythonImportsCheck = [ "alf" ];

  meta = with lib; {
    description = ''
      Agent Learning Framework (ALF) is a reinforcement learning framework
      emphasizing on the flexibility and easiness of implementing complex
      algorithms involving many different components
    '';
    homepage = "https://alf.readthedocs.io/en/latest/index.html";
    license = licenses.asl20;
    maintainers = with maintainers; [ breakds ];
  };
}
