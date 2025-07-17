import subprocess
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def run_script(script_path, img_path, extra_args=None):
    """
    Run a single Python script.

    :param script_path: The full path to the script to be executed
    :param img_path: The image path parameter
    :param extra_args: Additional arguments (as a dictionary)
    """
    command = ["python", script_path, "--img_path", img_path]
    
    if extra_args:
        for key, value in extra_args.items():
            command.extend([key, value])
    
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Output from {os.path.basename(script_path)}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {os.path.basename(script_path)}:")
        print(e.stderr)
        sys.exit(e.returncode)

def parse_arguments():
    """
    Parse command-line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run multiple Python scripts to process images.")
    parser.add_argument(
        '--img_path',
        type=str,
        required=True,
        help="Image path to be used for the --img_path parameter in all scripts."
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help="Whether to run scripts in parallel. Defaults to sequential execution."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    img_path = args.img_path
    run_in_parallel = args.parallel

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(current_dir, "eval")

    # Define the list of scripts to run, including the eval folder path
    scripts = [
        os.path.join(eval_dir, "PickScore.py"),
        os.path.join(eval_dir, "Aesthetics_score.py"),
        os.path.join(eval_dir, "clip_score.py"),
        os.path.join(eval_dir, "Image_reward.py"),
        os.path.join(eval_dir, "HPSv2_Eval.py")
    ]

    if run_in_parallel:
        # Run scripts in parallel
        with ThreadPoolExecutor(max_workers=len(scripts)) as executor:
            future_to_script = {}
            for script in scripts:
                if os.path.basename(script) == "HPSv2_Eval.py":
                    extra_args = {"--hps_version": "v2.0"}
                    future = executor.submit(run_script, script, img_path, extra_args)
                else:
                    future = executor.submit(run_script, script, img_path)
                future_to_script[future] = script
            
            for future in as_completed(future_to_script):
                script = future_to_script[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"{os.path.basename(script)} generated an exception: {exc}")
    else:
        # Run scripts sequentially
        for script in scripts:
            if os.path.basename(script) == "HPSv2_Eval.py":
                # Add additional arguments for HPSv2_Eval.py
                extra_args = {"--hps_version": "v2.0"}
                run_script(script, img_path, extra_args)
            else:
                # Other scripts only use the --img_path parameter
                run_script(script, img_path)
    
    print("All scripts have been executed.")

if __name__ == "__main__":
    main()