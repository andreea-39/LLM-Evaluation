import importlib
import os


def is_route_dir_abs(path):
    absPath = os.path.normpath(os.path.join(os.path.dirname(__file__), path))
    if os.path.isdir(absPath):
        if "route.py" in os.listdir(absPath):
            return True
    return False


for folder in filter(is_route_dir_abs, os.listdir(os.path.dirname(__file__))):
    if folder == "__pycache__":
        continue
    module = importlib.import_module("." + folder + ".route", package=__name__)
