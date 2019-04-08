
## Workflow

1. Clone

2. Install code style tools
```bash
pip install pre-commit
pip install cpplint
pip install pydocstyle
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

Make change to your code and test

5. Make pull request:
```bash
git push origin PR_change_name
```

## Coding stardard

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
