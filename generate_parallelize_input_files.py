import itertools
import os
import argparse

def generate_param_combinations(params_dict):
    """
    Generate all combinations of parameters based on provided dictionary values.
    """
    keys = params_dict.keys()
    values = (params_dict[key] if isinstance(params_dict[key], list) else [params_dict[key]] for key in keys)
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]

def create_input_files(output_dir, base_filename, param_combinations):
    """
    Create .txt input files for parallelize_main.sh, containing parameter combinations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(file_path, "w") as f:
        for params in param_combinations:
            f.write(",".join(str(params[key]) for key in params.keys()) + "\n")
    
    print(f"Generated input file: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate .txt files for parallelize_main.sh")
    parser.add_argument("--output_dir", type=str, default="parallel_inputs", help="Directory to save the generated .txt files.")
    parser.add_argument("--filename", type=str, default="input", help="Base name of the output .txt file.")
    args = parser.parse_args()
    
    # Define parameter values (modify as needed)
    params = {
        "seed_index": [1, 2, 3],
        "algorithm": ["gp", "gsgp", "slim"],
        "dataset": ["airfoil", "concrete"],
        "pop_size": [100],
        "n_iter": [1000],
        "n_elites": [1],
        "pressure": [4],
        "slim_crossover": ["default"],
        "p_inflate": [0.0],
        "p_crossover": [0.9],
        "p_mutation": [0.1],
        "torus_dim": [0],
        "pop_shape": ["10x10"],
        "radius": [0],
        "cmp_rate": [0.0],
        "run_id": [args.filename]
    }
    
    param_combinations = generate_param_combinations(params)
    create_input_files(args.output_dir, args.filename, param_combinations)

if __name__ == "__main__":
    main()

