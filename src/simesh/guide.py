"""Display the workflow guide shipped with this installed version."""

from importlib.resources import files


if __name__ == "__main__":
    print(files("simesh").joinpath("_guide.md").read_text(encoding="utf-8"), end="")
