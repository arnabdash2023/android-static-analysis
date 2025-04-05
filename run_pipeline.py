import subprocess
import os
import sys
import logging
from datetime import datetime
import argparse
import glob
import time
import getpass

# Configure logging with current date/time
current_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
current_user = getpass.getuser()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_run_{current_time}_{current_user}.log'),
        logging.StreamHandler()
    ]
)

class PipelineRunner:
    def __init__(self, base_dir=None, benign_dir=None, malware_dir=None, workers=5, save_interval=50):
        """
        Initialize the pipeline runner with configuration parameters
        
        Args:
            base_dir (str): Base working directory (defaults to script's directory)
            benign_dir (str): Directory containing benign APKs (defaults to 'benignSample')
            malware_dir (str): Directory containing malware APKs (defaults to 'malwareSample')
            workers (int): Number of worker processes
            save_interval (int): Save interval for feature extraction
        """
        # Default to the directory containing this script if base_dir not provided
        self.base_dir = base_dir if base_dir else os.path.dirname(os.path.abspath(__file__))
        self.benign_dir = benign_dir if benign_dir else os.path.join(self.base_dir, 'benignSample')
        self.malware_dir = malware_dir if malware_dir else os.path.join(self.base_dir, 'malwareSample')
        self.workers = workers
        self.save_interval = save_interval
        
        # Define script paths dynamically
        self.extract_script = os.path.join(self.base_dir, 'extract_apk_features.py')
        self.drop_script = os.path.join(self.base_dir, 'drop_irrelevant_features.py')
        self.preprocess_script = os.path.join(self.base_dir, 'android_malware_preprocessing.py')
        self.model_script = os.path.join(self.base_dir, 'model_comparison.py')
        
        # Define output files dynamically
        self.feature_file = os.path.join(self.base_dir, 'apk_features_updated.csv')
        self.cleaned_file = os.path.join(self.base_dir, 'cleaned_features.csv')

        # Add stage weights (based on timings)
        self.stage_weights = {
            "Feature Extraction": 40,
            "Feature Dropping": 10,
            "Preprocessing": 20,
            "Model Comparison": 30
        }
        self.current_progress = 0
        self.start_time = None

    def update_progress(self, stage_name, status="started"):
        """Update and display progress for the current stage"""
        if self.start_time is None:
            self.start_time = time.time()

        if status == "started":
            stage_msg = f"Starting {stage_name}..."
        elif status == "completed":
            self.current_progress += self.stage_weights[stage_name]
            stage_msg = f"{stage_name} completed"

        elapsed_time = time.time() - self.start_time
        progress_bar = "=" * (self.current_progress // 2) + ">" + " " * ((100 - self.current_progress) // 2)
        
        # Calculate estimated time remaining
        if self.current_progress > 0:
            eta = (elapsed_time * (100 - self.current_progress)) / self.current_progress
        else:
            eta = 0

        logging.info(
            f"[{progress_bar}] {self.current_progress}% | {stage_msg} | "
            f"Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s"
        )

    def check_file_exists(self, filepath, description):
        """Check if a file or directory exists and log the result"""
        if not os.path.exists(filepath):
            logging.error(f"{description} not found at {filepath}")
            return False
        logging.info(f"{description} found at {filepath}")
        return True

    def run_command(self, command, step_name):
        """Run a shell command with error handling and progress updates"""
        try:
            self.update_progress(step_name, "started")
            
            process = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.update_progress(step_name, "completed")
            
            if process.stderr:
                logging.warning(f"Warnings/Errors:\n{process.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error in {step_name}: {e}")
            logging.error(f"Command output: {e.output}")
            return False
        except KeyboardInterrupt:
            logging.warning(f"{step_name} interrupted by user")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in {step_name}: {e}")
            return False

    def run_pipeline(self):
        """Execute the full pipeline with progress tracking"""
        try:
            self.start_time = time.time()
            self.current_progress = 0
            
            logging.info(f"Pipeline started by user: {current_user}")
            logging.info(f"Start time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

            # Step 1: Feature Extraction
            extract_cmd = (
                f"python3 {self.extract_script} "
                f"--benign {self.benign_dir} "
                f"--malware {self.malware_dir} "
                f"--output {self.feature_file} "
                f"--workers {self.workers} "
                f"--save-interval {self.save_interval} "
                f"> /dev/null 2>&1"
            )
            
            if not all([
                self.check_file_exists(self.extract_script, "Feature extraction script"),
                self.check_file_exists(self.benign_dir, "Benign APK directory"),
                self.check_file_exists(self.malware_dir, "Malware APK directory")
            ]):
                return False
                
            if not self.run_command(extract_cmd, "Feature Extraction"):
                return False

            # Step 2: Drop Irrelevant Features
            drop_cmd = (
                f"python3 {self.drop_script} "
                f"--input {self.feature_file} "
                f"--output {self.cleaned_file}"
            )
            
            if not self.check_file_exists(self.drop_script, "Drop features script"):
                return False
                
            if not os.path.exists(self.feature_file):
                logging.error(f"Feature file {self.feature_file} not found after extraction")
                return False
                
            if not self.run_command(drop_cmd, "Feature Dropping"):
                return False

            # Step 3: Preprocessing
            preprocess_cmd = (
                f"python3 {self.preprocess_script} "
                f"--input {self.cleaned_file}"
            )
            
            if not self.check_file_exists(self.preprocess_script, "Preprocessing script"):
                return False
                
            if not os.path.exists(self.cleaned_file):
                logging.error(f"Cleaned feature file {self.cleaned_file} not found after feature dropping")
                return False
                
            if not self.run_command(preprocess_cmd, "Preprocessing"):
                return False

            # Step 4: Model Comparison
            model_cmd = f"python3 {self.model_script}"
            
            if not self.check_file_exists(self.model_script, "Model comparison script"):
                return False
                
            if not self.run_command(model_cmd, "Model Comparison"):
                return False

            end_time = time.time()
            total_time = end_time - self.start_time
            logging.info(f"Pipeline completed successfully!")
            logging.info(f"End time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Total execution time: {total_time:.1f}s")
            return True

        except KeyboardInterrupt:
            logging.warning("Pipeline interrupted by user. Partial results may be available.")
            return False

    def resume_pipeline(self):
        """Resume the pipeline from the last successful step with progress tracking"""
        logging.info("Attempting to resume pipeline from last successful step...")
        self.start_time = time.time()
        
        # Check if feature extraction was completed
        if os.path.exists(self.feature_file):
            logging.info(f"Feature file {self.feature_file} exists, skipping Feature Extraction")
            self.current_progress = self.stage_weights["Feature Extraction"]
        else:
            logging.info("Resuming with Feature Extraction")
            return self.run_pipeline()

        # Check if feature dropping was completed
        if os.path.exists(self.cleaned_file):
            logging.info(f"Cleaned feature file {self.cleaned_file} exists, skipping Feature Dropping")
            self.current_progress += self.stage_weights["Feature Dropping"]
        else:
            drop_cmd = (
                f"python3 {self.drop_script} "
                f"--input {self.feature_file} "
                f"--output {self.cleaned_file}"
            )
            if not self.run_command(drop_cmd, "Feature Dropping"):
                return False

        # Check if preprocessing was completed
        preprocessed_dirs = glob.glob(os.path.join(self.base_dir, 'preprocessed_data_*'))
        if preprocessed_dirs:
            latest_dir = max(preprocessed_dirs, key=os.path.getctime)
            logging.info(f"Preprocessed data found in {latest_dir}, skipping Preprocessing")
            self.current_progress += self.stage_weights["Preprocessing"]
        else:
            preprocess_cmd = (
                f"python3 {self.preprocess_script} "
                f"--input {self.cleaned_file}"
            )
            if not self.run_command(preprocess_cmd, "Preprocessing"):
                return False

        # Run model comparison
        model_cmd = f"python3 {self.model_script}"
        if not self.run_command(model_cmd, "Model Comparison"):
            return False

        end_time = time.time()
        total_time = end_time - self.start_time
        logging.info("Pipeline resumed and completed successfully!")
        logging.info(f"End time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total execution time: {total_time:.1f}s")
        return True

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Android Malware Analysis Pipeline')
    parser.add_argument('--base-dir', help='Base directory containing scripts (default: script directory)')
    parser.add_argument('--benign-dir', help='Directory containing benign APK samples (default: benignSample)')
    parser.add_argument('--malware-dir', help='Directory containing malware APK samples (default: malwareSample)')
    parser.add_argument('--workers', type=int, default=5, help='Number of worker processes for feature extraction')
    parser.add_argument('--save-interval', type=int, default=50, help='Save interval for feature extraction')
    parser.add_argument('--resume', action='store_true', help='Resume pipeline from last successful step')
    parser.add_argument('--clean', type=int, default=0, help='Clean the out directory')

    args = parser.parse_args()

    if args.clean:
        logging.info("Cleaning output directories and files...")
        # Remove directories matching 'preprocessed*' and 'trainModel/'
        for dir_pattern in ['preprocessed*', 'trainModel']:
            for dir_path in glob.glob(dir_pattern):
                try:
                    import shutil
                    shutil.rmtree(dir_path)
                    logging.info(f"Removed directory: {dir_path}")
                except OSError as e:
                    logging.error(f"Error removing {dir_path}: {e.strerror}")

        # Remove log and csv files
        files = glob.glob('*.log') + glob.glob('*.csv')
        for f in files:
            try:
                os.remove(f)
                logging.info(f"Removed file: {f}")
            except OSError as e:
                logging.error(f"Error removing {f}: {e.strerror}")

        return True

    # Create pipeline runner instance
    runner = PipelineRunner(
        base_dir=args.base_dir,
        benign_dir=args.benign_dir,
        malware_dir=args.malware_dir,
        workers=args.workers,
        save_interval=args.save_interval
    )

    # Run or resume the pipeline
    if args.resume:
        success = runner.resume_pipeline()
    else:
        success = runner.run_pipeline()
    
    if not success:
        logging.error("Pipeline failed or was interrupted. Check logs for details.")
        sys.exit(1)
    else:
        logging.info("Pipeline executed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()