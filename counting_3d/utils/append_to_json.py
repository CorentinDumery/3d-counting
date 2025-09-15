import json
import os


def append_to_json(path, key, value):
    data = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)

    data[key] = value

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
