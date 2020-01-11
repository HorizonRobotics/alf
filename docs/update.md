# How to update third-party software?

ALF depends on a lot of third party software. The folloing is the workflow for
updating 3rd party software. We will use ALF_ROOT to represent the directory of
ALF source root, ALF_VERSION to represent the name of the new version (e.g. 0.1.0 or latest)

0. Prepare settings:
   ```bash
   ALF_VERSION=...
   ALF_ROOT=...
   ```
1. Install the desired version of 3rd party software locally and test using
   the following commands:
   ```bash
   cd $ALF_ROOT
   python -m unittest discover -s alf -p "*_test.py" -v
   ```
2. Modify [setup.py](../setup.py), [.ci-cd/requirements.txt](../.ci-cd/requirements.txt) and [.ci-cd/Dockerfile.cpu](../.ci-cd/Dockerfile.cpu). If you are updating tensorflow, you need update Dockerfile.cpu to use the appropriate
   tensorflow docker image.
3. Build the new docker image:
   ```bash
   cd $ALF_ROOT/.ci-cd
   docker build -t horizonrobotics/alf:$ALF_VERSION -f Dockerfile.cpu .
   ```
4. Test the image locally:
   ```bash
   docker run -v $ALF_ROOT:/ALF -w /ALF/ -e PYTHONPATH=/ALF -it horizonrobotics/alf:$ALF_VERSION /ALF/.ci-cd/build.sh check_style
   docker run -v $ALF_ROOT:/ALF -w /ALF/ -e PYTHONPATH=/ALF -it horizonrobotics/alf:$ALF_VERSION /ALF/.ci-cd/build.sh test
   ```
5. Push the docker image to docker hub. Note that you need to have an account
   with the necessary access permission to horizonrobotics/alf on hub.docker.com.
   ```bash
   docker login
   docker push horizonrobotics/alf:$ALF_VERSION
   docker logout
   ```
6. Update [.travis.yml](../.travis.yml). Change `horizonrobotics/alf:xxx` to the new docker image
version `horizonrobotics/alf:ALF_VERSION`
7. Send your change to github for code review.
