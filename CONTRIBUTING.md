
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
