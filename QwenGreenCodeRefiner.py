import os
import shutil
import logging
import time
import requests
import json
from pathlib import Path
from typing import List, Set
from dotenv import load_dotenv
from tqdm import tqdm
import csv
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('code_refiner.log'),
        logging.StreamHandler()
    ]
)

env_path = os.path.abspath(".env")
BASE_DIR = os.path.dirname(env_path)
RESULT_DIR = os.path.join(BASE_DIR, 'Result')

class MetricsTracker:
    def __init__(self, base_dir):
        """Initialize metrics tracking."""
        self.start_time = time.time()
        self.files_modified = 0
        self.loc_by_extension = defaultdict(int)
        self.base_dir = base_dir
        
    def count_loc(self, file_path: Path) -> int:
        """Count lines of code in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            logging.error(f"Error counting LOC for {file_path}: {e}")
            return 0
            
    def track_file(self, file_path: Path):
        """Track metrics for a single file."""
        self.files_modified += 1
        loc = self.count_loc(file_path)
        ext = file_path.suffix[1:] if file_path.suffix.startswith('.') else file_path.suffix
        self.loc_by_extension[ext] += loc
        
    def get_processing_time(self) -> float:
        """Get processing time in minutes."""
        return (time.time() - self.start_time) / 60

class MetricsHandler:
    @staticmethod
    def load_historical_data(csv_path: Path) -> dict:
        """Load historical data from existing CSV file."""
        historical_data = {}
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    historical_section = False
                    for row in reader:
                        if not row:
                            continue
                        if row[0] == '=== Historical Overview ===':
                            historical_section = True
                            continue
                        if historical_section and len(row) == 2:
                            metric, value = row[0], row[1]
                            # Handle different value formats
                            if 'LOC' in value:
                                # Extract numeric value before 'LOC'
                                historical_data[metric] = int(value.split()[0])
                            elif '.' in value:
                                # Handle floating point values
                                historical_data[metric] = float(value)
                            else:
                                # Handle integer values
                                historical_data[metric] = int(value)
            except Exception as e:
                logging.warning(f"Error loading historical data: {e}")
        return historical_data

    @staticmethod
    def update_final_overview(metrics_tracker, project_path: Path):
        """Update the final overview CSV with fresh and historical data."""
        result_dir = project_path / 'Result'  # Use the provided project_path
        result_dir.mkdir(parents=True, exist_ok=True)
        csv_path = result_dir / 'final_overview.csv'
        
        # Load historical data
        historical_data = MetricsHandler.load_historical_data(csv_path)
        
        # Get current run metrics
        current_metrics = MetricsHandler.get_current_run_metrics(metrics_tracker)
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers
                writer.writerow(['Metric', 'Value'])
                writer.writerow([])  # Empty row for separation
                
                # Fresh Details Section
                writer.writerow(['=== Fresh Details (Last Run) ==='])
                for metric, value in current_metrics.items():
                    writer.writerow([metric, value])
                
                writer.writerow([])  # Empty row for separation
                
                # Historical Overview Section
                writer.writerow(['=== Historical Overview ==='])
                historical_metrics = MetricsHandler.combine_metrics(historical_data, current_metrics)
                for metric, value in historical_metrics.items():
                    writer.writerow([metric, value])
                    
            logging.info(f"Updated final overview at: {csv_path}")
        except Exception as e:
            logging.error(f"Failed to update final overview: {e}")

    @staticmethod
    def get_current_run_metrics(metrics_tracker) -> dict:
        """Get metrics for the current run."""
        metrics = {
            'Total Files Modified (Last run)': metrics_tracker.files_modified,
            'Total LOC Converted (Last run)': sum(metrics_tracker.loc_by_extension.values()),
            'Total Time (minutes) (Last run)': round(metrics_tracker.get_processing_time(), 2)
        }
        
        # Add per-extension metrics
        for ext, loc in metrics_tracker.loc_by_extension.items():
            metrics[f'.{ext} Files (Last run)'] = f"{loc} LOC"
        
        return metrics

    @staticmethod
    def combine_metrics(historical, current):
        """Combine historical and current metrics."""
        combined = {}
        
        # Base metrics
        combined['Total Files Modified'] = (
            historical.get('Total Files Modified', 0) +
            current.get('Total Files Modified (Last run)', 0)
        )
        
        combined['Total LOC Converted'] = (
            historical.get('Total LOC Converted', 0) +
            current.get('Total LOC Converted (Last run)', 0)
        )
        
        combined['Total Time (minutes)'] = round(
            historical.get('Total Time (minutes)', 0) +
            current.get('Total Time (minutes) (Last run)', 0),
            2
        )
        
        # Combine extension-specific metrics
        extension_metrics = {}
        for key, value in current.items():
            if key.endswith('Files (Last run)'):
                ext = key.split()[0]
                extension_metrics[ext] = int(value.split()[0])
        
        for key, historical_value in historical.items():
            if isinstance(historical_value, str) and 'LOC' in historical_value:
                ext = f'.{key.split()[0]}'
                hist_loc = int(historical_value.split()[0])
                current_loc = extension_metrics.get(ext, 0)
                combined[f'{ext} Files'] = f"{hist_loc + current_loc} LOC"
        
        return combined

def ensure_result_directory():
    """Ensure the Result directory exists."""
    if not os.path.exists(RESULT_DIR):
        try:
            os.makedirs(RESULT_DIR)
            logging.info(f"Created Result directory at: {RESULT_DIR}")
        except Exception as e:
            logging.error(f"Failed to create Result directory: {e}")
            raise

def ensure_csv_exists():
    """Ensure the CSV file exists and has the correct headers."""
    ensure_result_directory()  # Make sure the Result directory exists
    csv_path = os.path.join(RESULT_DIR, 'modification_overview.csv')
    
    if not os.path.exists(csv_path):
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['File Name', 'Modification Timestamp', 'Changes', 'Next Steps'])
                writer.writeheader()
            logging.info(f"Created modification_overview.csv at: {csv_path}")
        except Exception as e:
            logging.error(f"Failed to create CSV file: {e}")
            raise

def extract_section(content, start_marker, end_marker):
    """Extract content between specific markers."""
    try:
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return None
        
        start_idx += len(start_marker)
        end_idx = content.find(end_marker, start_idx)
        
        if end_idx == -1:
            # If no end marker, take till the end
            section = content[start_idx:].strip()
        else:
            section = content[start_idx:end_idx].strip()
        
        # Extract the first meaningful line if multiple lines exist
        lines = [line.strip('- *•').strip() for line in section.split('\n') 
                 if line.strip('- *•').strip() and 
                 not any(marker in line.lower() for marker in ['here is', 'summary', 'changes made'])]
        
        return lines[0] if lines else None
    except Exception as e:
        logging.warning(f"Error extracting section: {e}")
        return None

def log_modifications(file_name, changes, next_steps):
    """Log modifications to the CSV file."""
    csv_path = os.path.join(RESULT_DIR, 'modification_overview.csv')
    ensure_csv_exists()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['File Name', 'Modification Timestamp', 'Changes', 'Next Steps'])
            writer.writerow({
                'File Name': file_name,
                'Modification Timestamp': timestamp,
                'Changes': changes,
                'Next Steps': next_steps
            })
        logging.info(f"Logged modifications for file: {file_name}")
    except Exception as e:
        logging.error(f"Failed to log modifications: {e}")

def extract_changes_summary(response):
    """Extract changes summary from the assistant's response."""
    try:
        if not response or not isinstance(response, list) or not response[0].get('generated_text'):
            return None, None
            
        content = response[0]['generated_text']
        
        # Extract changes
        changes = extract_section(content, 'CHANGES_START', 'CHANGES_END')
        
        # Extract next steps
        next_steps = extract_section(content, 'NEXT_STEPS_START', 'NEXT_STEPS_END')
        
        return (changes or "No specific changes provided", 
                next_steps or "No specific next steps identified")
        
    except Exception as e:
        logging.warning(f"Error extracting changes summary: {e}")
        return "Error extracting changes", "Error identifying next steps"

class CodeRefiner:    
    def __init__(self):
        """Initialize the CodeRefiner with model from environment variables."""
        self.metrics_tracker = MetricsTracker(BASE_DIR)
        self.load_environment()
        self.setup_api()

    def parse_extensions(self, extensions_str: str) -> Set[str]:
        """Parse file extensions from string to set."""
        try:
            # Remove spaces and split by comma
            extensions = [ext.strip() for ext in extensions_str.split(',')]
            # Ensure all extensions start with a dot
            extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
            return set(extensions)
        except Exception as e:
            logging.error(f"Error parsing file extensions: {str(e)}")
            raise

    def get_default_models(self) -> dict:
        """Return default models configuration."""
        return {
            "qwen-7b-chat": "Qwen/Qwen-7B-Chat",
            "qwen-14b-chat": "Qwen/Qwen-14B-Chat",
            "qwen-1.5b": "Qwen/Qwen-1_8B-Chat",
            "qwen-coder": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "qwen-32b": "Qwen/Qwen2.5-Coder-32B-Instruct"
        }
   
   
    def load_environment(self) -> None:
        """Load and validate environment variables."""
        try:
            def parse_env_file(file_path: str) -> dict:
                env_vars = {}
                current_key = None
                current_value = []
                
                with open(file_path) as f:
                    lines = f.readlines()
                    
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        i += 1
                        continue
                    
                    # Check for new variable definition
                    if '=' in line and not current_key:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Check if this is the start of a multi-line string
                        if value.startswith('"""') and not value.endswith('"""'):
                            current_key = key
                            current_value = [value.lstrip('"')]
                        else:
                            # Single line value - remove quotes if present
                            if (value.startswith('"""') and value.endswith('"""')):
                                value = value[3:-3]
                            elif (value.startswith('"') and value.endswith('"')):
                                value = value[1:-1]
                            env_vars[key] = value
                    
                    # Continue collecting multi-line value
                    elif current_key:
                        if line.endswith('"""'):
                            current_value.append(line.rstrip('"'))
                            env_vars[current_key] = '\n'.join(current_value)
                            current_key = None
                            current_value = []
                        else:
                            current_value.append(line)
                    
                    i += 1
                
                return env_vars

            # Load and parse environment file
            if os.path.exists(env_path):
                env_vars = parse_env_file(env_path)
                # Set environment variables
                for key, value in env_vars.items():
                    os.environ[key] = value
            else:
                logging.warning(f"Environment file {env_path} not found. Using default values.")
                env_vars = {}
            
            # Load basic configurations
            self.project_path = BASE_DIR
            self.prompt = env_vars.get('PROMPT_1') or os.getenv('PROMPT_1')
            # Add test case prompt
            self.test_prompt = env_vars.get('PROMPT_GENERATE_TESTCASES') or os.getenv('PROMPT_GENERATE_TESTCASES', 
                'Create testcase functions for the given code. Provide only the code without any markdown, comments, or explanations.')
            self.hf_token = env_vars.get('HF_TOKEN') or os.getenv('HF_TOKEN')

            # Test suite directory names
            self.src_test_suite = 'SRC-TestSuite'
            self.greencode_test_suite = 'GreenCode-TestSuite'
            
            # Load API URL
            self.api_base_url = env_vars.get('API_URL') or os.getenv('API_URL', 'https://api-inference.huggingface.co/models/')
            
            # Load and parse file extensions
            extensions_str = env_vars.get('QWEN_FILE_EXTENSIONS') or os.getenv('QWEN_FILE_EXTENSIONS', '.py,.java,.cpp,.cs,.js,.ts')
            self.supported_extensions = self.parse_extensions(extensions_str)
            
            # Load excluded files
            excluded_files_str = env_vars.get('EXCLUDED_FILES') or os.getenv('EXCLUDED_FILES', '')
            self.excluded_files = [f.strip() for f in excluded_files_str.split(',') if f.strip()]
            
            # Use default models configuration
            self.available_models = self.get_default_models()
            
            # Get selected model from environment
            self.model_key = (env_vars.get('AZURE_MODEL') or os.getenv('AZURE_MODEL', '')).lower()
            if not self.model_key or self.model_key not in self.available_models:
                available_models = list(self.available_models.keys())
                raise ValueError(f"Invalid or missing model selection in .env file. Choose from: {available_models}")
            
            # Validate required variables
            if not all([self.project_path, self.prompt, self.hf_token]):
                missing = []
                if not self.project_path: missing.append('PROJECT_PATH')
                if not self.prompt: missing.append('PROMPT_1')
                if not self.hf_token: missing.append('HF_TOKEN')
                raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
                
            self.project_path = Path(self.project_path)
            if not self.project_path.exists():
                raise ValueError(f"Project path does not exist: {self.project_path}")
                
            # Log configurations
            logging.info(f"Environment loaded successfully")
            logging.info(f"Selected model: {self.model_key}")
            logging.info(f"Project path: {self.project_path}")
            logging.info(f"Supported extensions: {self.supported_extensions}")
            logging.info(f"API URL: {self.api_base_url}")
            logging.info(f"Excluded files: {self.excluded_files}")
            
        except Exception as e:
            logging.error(f"Error loading environment: {str(e)}")
            raise
   
    def setup_api(self) -> None:
        """Setup the API connection."""
        try:
            self.model_url = self.api_base_url + self.available_models[self.model_key]
            self.headers = {"Authorization": f"Bearer {self.hf_token}"}
            logging.info(f"API setup completed for model: {self.available_models[self.model_key]}")
        except Exception as e:
            logging.error(f"Error setting up API: {str(e)}")
            raise

    def query_api(self, payload: dict) -> dict:
        """Send request to Hugging Face API."""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.model_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"API request failed after {max_retries} attempts: {str(e)}")
                logging.warning(f"API request attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    def refine_code(self, code: str, file_path: Path) -> str:
        """Refine code using the Hugging Face API."""
        try:
            # Prepare the prompt
            prompt = f"{self.prompt}\n\nCode:\n{code}\n\nRefined Code:"
            
            # Prepare the payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
            
            # Query the API
            response = self.query_api(payload)
            
            # Extract the refined code
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')
                refined_code = generated_text.split("Refined Code:")[-1].strip()
                
                # Extract changes and next steps
                changes, next_steps = extract_changes_summary(response)
                log_modifications(file_path.name, changes, next_steps)
                
                return refined_code
            else:
                raise ValueError("Unexpected API response format")
            
        except Exception as e:
            logging.error(f"Error refining code: {str(e)}")
            raise

    def process_file(self, file_path: Path) -> None:
        """Process a single code file."""
        try:
            # Skip excluded files
            if file_path.name in self.excluded_files:
                logging.info(f"Skipping excluded file: {file_path.name}")
                return
                
            # Read the original code
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
                
            # Refine the code (pass file_path to refine_code)
            refined_code = self.refine_code(original_code, file_path)
            
            # Calculate relative path
            relative_path = file_path.relative_to(self.project_path)
            output_file = self.output_path / relative_path
            
            # Create necessary directories
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write refined code
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(refined_code)
                
            logging.info(f"Successfully processed: {relative_path}")

            # Track metrics for the processed file
            self.metrics_tracker.track_file(output_file)
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def setup_output_directory(self) -> None:
        """Setup the GreenCode output directory."""
        try:
            self.output_path = self.project_path / 'GreenCode'
            
            if self.output_path.exists():
                logging.info("Removing existing GreenCode directory...")
                shutil.rmtree(self.output_path)
                
            self.output_path.mkdir(parents=True)
            logging.info(f"Created output directory: {self.output_path}")
            
        except Exception as e:
            logging.error(f"Error setting up output directory: {str(e)}")
            raise

    def get_code_files(self) -> List[Path]:
        """Recursively find all supported code files in the project."""
        try:
            code_files = []
            total_files = 0
            processed_files = 0
            skipped_files = 0
            
            # Log the start of file scanning
            logging.info("Starting to scan for code files...")
            logging.info(f"Looking for files with extensions: {self.supported_extensions}")
            
            for file_path in self.project_path.rglob('*'):
                total_files += 1
                
                # Skip directories and handle files only
                if not file_path.is_file():
                    continue
                    
                # Check if file is in SRC-TestSuite folder or other excluded conditions
                if (file_path.suffix in self.supported_extensions and 
                    'GreenCode' not in file_path.parts and
                    self.src_test_suite not in file_path.parts and  # Exclude SRC-TestSuite folder
                    file_path.name not in self.excluded_files):
                    code_files.append(file_path)
                    processed_files += 1
                else:
                    skipped_files += 1
            
            # Log detailed statistics
            logging.info(f"File scanning completed:")
            logging.info(f"Total files scanned: {total_files}")
            logging.info(f"Files to process: {processed_files}")
            logging.info(f"Files skipped: {skipped_files}")
            logging.info(f"Supported extensions found: {set(f.suffix for f in code_files)}")
            
            # Additional validation
            if not code_files:
                logging.warning("No matching code files found in the project directory!")
            
            return code_files
            
        except Exception as e:
            logging.error(f"Error finding code files: {str(e)}")
            raise

    def is_test_file(self, file_path: Path) -> bool:
        """Check if a file is a test file."""
        test_indicators = ['test', 'spec', 'Tests']
        file_stem = file_path.stem.lower()
        return any(indicator.lower() in file_stem for indicator in test_indicators)

    def get_test_file_path(self, source_file: Path, base_dir: Path, test_suite_dir: Path) -> Path:
        """Generate the path for the test file."""
        # Get relative path from base directory
        relative_path = source_file.relative_to(base_dir)
        
        # Create test file path
        test_file_name = f"{source_file.stem}Test{source_file.suffix}"
        test_file_path = test_suite_dir / relative_path.parent / test_file_name
        
        return test_file_path

    def existing_test_file(self, source_file: Path, test_suite_dir: Path) -> bool:
        """Check if a test file already exists in the test suite directory."""
        potential_test_names = [
            f"{source_file.stem}Test{source_file.suffix}",
            f"test_{source_file.stem}{source_file.suffix}",
            f"{source_file.stem}_test{source_file.suffix}",
            f"{source_file.stem}.test{source_file.suffix}"
        ]
        
        # Get relative path without filename
        relative_dir = source_file.parent.relative_to(self.project_path)
        test_dir = test_suite_dir / relative_dir
        
        # Check in the corresponding test suite directory
        for test_name in potential_test_names:
            if (test_dir / test_name).exists():
                return True
        
        return False


    def generate_test_case(self, code: str) -> str:
        """Generate test cases using the API."""
        try:
            # Prepare the prompt
            prompt = f"{self.test_prompt}\n\nCode:\n{code}"
            
            # Prepare the payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
            
            # Query the API
            response = self.query_api(payload)
            
            # Extract the test code
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')
                test_code = generated_text.split("Code:")[-1].strip()
                return test_code
            else:
                raise ValueError("Unexpected API response format")
            
        except Exception as e:
            logging.error(f"Error generating test case: {str(e)}")
            raise

    def process_tests_for_directory(self, source_dir: Path, test_suite_dir: Path) -> None:
        """Process test cases for all files in a directory."""
        try:
            # Create test suite directory
            test_suite_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect files that need test cases
            code_files = []
            for file_path in source_dir.rglob('*'):
                # Determine if we're processing GreenCode directory
                is_greencode = 'GreenCode' in source_dir.parts
                
                if (file_path.suffix in self.supported_extensions and 
                    not self.is_test_file(file_path) and 
                    not self.existing_test_file(file_path, test_suite_dir) and
                    (is_greencode or 'GreenCode' not in file_path.parts) and  # Modified condition
                    'TestSuite' not in file_path.parts and
                    file_path.name not in self.excluded_files):
                    code_files.append(file_path)
            
            logging.info(f"Found {len(code_files)} files requiring test cases in {source_dir}")
            
            # Add more detailed logging
            if not code_files:
                logging.warning(f"No files found matching criteria in {source_dir}. Check file extensions and exclusion rules.")
                for file_path in source_dir.rglob('*'):
                    if file_path.is_file():
                        logging.debug(f"Found file: {file_path}, Extension: {file_path.suffix}")
            
            for file_path in tqdm(code_files, desc=f"Generating tests for {source_dir.name}"):
                try:
                    # Read the source code
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    # Generate test case
                    test_code = self.generate_test_case(source_code)
                    
                    # Get test file path
                    test_file_path = self.get_test_file_path(file_path, source_dir, test_suite_dir)
                    
                    # Create directory if it doesn't exist
                    test_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write test file
                    with open(test_file_path, 'w', encoding='utf-8') as f:
                        f.write(test_code)
                    
                    logging.info(f"Generated test case for: {file_path.relative_to(source_dir)}")
                    
                except Exception as e:
                    logging.error(f"Error processing tests for {file_path}: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error processing tests for directory {source_dir}: {str(e)}")
            raise

    def process_greencode_tests(self) -> None:
        """Process test cases for GreenCode directory."""
        try:
            # Create test suite directory inside GreenCode
            test_suite_dir = self.output_path / self.greencode_test_suite
            
            # Process only files in GreenCode directory
            self.process_tests_for_directory(self.output_path, test_suite_dir)
            
        except Exception as e:
            logging.error(f"Error processing GreenCode tests: {str(e)}")
            raise

    def process_source_tests(self) -> None:
        """Process test cases for source directory."""
        try:
            # Create test suite directory in project root
            test_suite_dir = self.project_path / self.src_test_suite
            
            # Process only files in source directory
            self.process_tests_for_directory(self.project_path, test_suite_dir)
            
        except Exception as e:
            logging.error(f"Error processing source tests: {str(e)}")
            raise


    def run(self) -> None:
        """Main execution method."""
        try:
            print(f"Starting code refinement and test generation process...")
            
            # Ensure Result directory exists
            ensure_result_directory()

            # First phase: Code refinement
            self.setup_output_directory()
            code_files = self.get_code_files()
            
            if not code_files:
                print("No files to process. Please check your QWEN_FILE_EXTENSIONS configuration.")
                return
                
            # Process code refinement
            for file_path in tqdm(code_files, desc="Processing files for optimization"):
                try:
                    print(f"\nOptimizing: {file_path.name}")
                    self.process_file(file_path)
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"Skipping file {file_path} due to error: {str(e)}")
                    continue
            
            logging.info("Code refinement completed successfully")
            
            # Second phase: Test case generation
            print("\nStarting test case generation...")
            
            # Generate tests for root directory first
            print("\nGenerating tests for original source files...")
            self.process_source_tests()
            
            # Generate tests for GreenCode directory
            print("\nGenerating tests for optimized files...")
            self.process_greencode_tests()

            # Track metrics for generated test files
            self.track_test_files()

            # Update final overview
            # self.metrics_tracker.update_final_overview()
            
            # Update final overview
            MetricsHandler.update_final_overview(self.metrics_tracker, self.project_path)
            
            logging.info("Test case generation completed successfully")
            print("\nCode refinement and test generation process completed!")
            
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            raise

    def track_test_files(self):
        """Track metrics for generated test files."""
        test_directories = [
            self.project_path / self.src_test_suite,
            self.output_path / self.greencode_test_suite
        ]
        
        for test_dir in test_directories:
            if test_dir.exists():
                for test_file in test_dir.rglob('*'):
                    if test_file.is_file() and test_file.suffix in self.supported_extensions:
                        self.metrics_tracker.track_file(test_file)

if __name__ == "__main__":
    try:
        refiner = CodeRefiner()
        refiner.run()
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
