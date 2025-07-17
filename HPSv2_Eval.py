import argparse
import hpsv2
import os

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Evaluate HPSv2 model using specified image path.")
    
    # Add the img_path parameter
    parser.add_argument(
        '--img_path',
        type=str,
        required=True,
        help='Path to the image directory for evaluation.'
    )
    
    # Optional parameter hps_version, default is "v2.0"
    parser.add_argument(
        '--hps_version',
        type=str,
        default="v2.0",
        help='Version of HPS to use for evaluation. Default is "v2.0".'
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Get the parameter values
    img_path = args.img_path
    hps_version = args.hps_version
    
    # Check if img_path exists
    if not os.path.exists(img_path):
        print(f"Error: The specified image path '{img_path}' does not exist.")
        return
    
    # Call the hpsv2.evaluate function
    try:
        hpsv2.evaluate(img_path, hps_version=hps_version)
        print(f"Evaluation completed successfully for path: {img_path} with HPS version: {hps_version}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()