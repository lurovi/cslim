import itertools
import os
import argparse
import json


def generate_param_combinations(params_dict):
    """
    Generate all combinations of parameters based on provided dictionary values.
    """
    keys = params_dict.keys()
    values = (params_dict[key] if isinstance(params_dict[key], list) else [params_dict[key]] for key in keys)
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def create_input_file(output_file, param_combinations):
    """
    Create a .txt input file for parallelize_main.sh, containing parameter combinations.
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        for params in param_combinations:
            f.write(",".join(str(params[key]) for key in params.keys()) + "," + output_file.replace(".txt", "") + "\n")

    print(f"Generated input file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate .txt file for parallelize_main.sh")
    parser.add_argument("--json_path", type=str, help="Path to the JSON file containing parameters, extension included.")
    args = parser.parse_args()

    # Derive output .txt filename from json_path
    json_dir, json_filename = os.path.split(args.json_path)
    txt_filename = os.path.splitext(json_filename)[0] + ".txt"
    output_file = os.path.join(json_dir, txt_filename)

    # Load parameters from JSON file
    with open(args.json_path, "r") as f:
        params = json.load(f)

    param_combinations = generate_param_combinations(params)
    create_input_file(output_file, param_combinations)


if __name__ == "__main__":
    main()
