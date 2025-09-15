import json
import argparse
import os

from utils.append_to_json import append_to_json

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-volume", type=float, required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--exp-name", required=True)
    args = parser.parse_args()

    # Path to results.json
    json_path = os.path.join(args.path, "results.json")

    with open(json_path, "r") as f:
        data = json.load(f)

        if "volume_in_cm3" not in data.keys():
            raise ValueError("Volume not in keys")

        if "volume_usage_" + args.exp_name not in data.keys():
            raise ValueError("Volume occupation not in keys")

        N = (
            data["volume_in_cm3"]
            * data["volume_usage_" + args.exp_name]
            / args.unit_volume
        )

    append_to_json(json_path, "N_" + args.exp_name, N)
    append_to_json(json_path, "N_int_" + args.exp_name, round(N))

    print("N", N)
