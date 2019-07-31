
## Workflow

1. Clone

2. Install code style tools
```bash
pip install pre-commit==1.17.0
pip install cpplint==1.4.4
pip install pydocstyle==4.0.0
pip install pylint==2.3.1
pip install yapf==0.28.0
sudo apt install clang-format
```

3. At your local repo root, run
```bash
pre-commit install
```

4. Make local changes
```bash
git co -b PR_change_name origin/master
```

Make change to your code and test. You can run all the existing unittests
by the following command:
```bash
python -m unittest discover -s alf -p "*_test.py" -v
```

Then commit your change to the local branch using `git commit`

5. Make pull request:
```bash
git push origin PR_change_name
```
6. Change your code based on review comments. The new change should be added
as NEW commit to your previous commits. Do not use `--amend` option for the 
commit, because then you will have to use '-f' option to push your change to
github and review will be more difficult because the new change cannot
be separated from previous change. For the same reason, if you need to incorporate
the latest code from master, please use `git pull` instead of `git pull --rebase`.

## Coding standard

We follow the coding style http://google.github.io/styleguide/pyguide.html. 
And please comment all the public functions with the following style:
```python
def func(a, b):
    """Short summary of the function

    Detailed explanation of the function. Including math equations and
    references. The explanation should be detail enough for the user to have a
    clear understanding of its function without reading its implementation.

    Args:
        a (type of a): purpose
        b (type of b): purpose
    Returns:
        return value1 (type 1): purpose
        return value2 (type 1): purpose
    """
```
