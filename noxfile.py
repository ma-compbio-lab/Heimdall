import nox


@nox.session
def flake8(session):
    session.install(
        "flake8",
        "flake8-absolute-import",
        "flake8-bugbear",
        "flake8-builtins",
        "flake8-colors",
        "flake8-commas",
        "flake8-comprehensions",
        # "flake8-docstrings",
        "flake8-import-order",
        "flake8-pyproject",
        "flake8-use-fstring",
        "pep8-naming",
    )
    session.run("flake8", "Heimdall/")
