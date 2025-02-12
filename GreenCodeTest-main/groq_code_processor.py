import os
import re
import json
import time
import logging
import shutil
import csv
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import tenacity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('refinement.log'), logging.StreamHandler()]
)

# Load .env configuration
env_path = os.path.abspath(".env")
load_dotenv(dotenv_path=env_path, override=True)
PROJECT_PATH = os.path.dirname(env_path)

# Configuration
EXCLUDED_FILES = [
    'GreenCodeRefiner.py', 'RefinerFunction.py', 'server_emissions.py',
    'track_emissions.py', 'report_template.html', 'details_template.html',
    'emissions_report.html', 'details_report.html', 'last_run_details_template.html',
    'last_run_report_template.html', 'server_report.html', 'AzureMarketplace.py',
    'details_server_template.html', 'recommendations_template.html',
    'code_refiner.py', 'recommendations_report.html', 'groq_code_processor.py', 'mul_server_emissions.py'
]

FILE_EXTENSIONS = ('.py', '.java', '.js', '.ts')
TEST_SUFFIX = 'Test'

# Directories
GREEN_CODE_DIR = os.path.join(PROJECT_PATH, 'GreenCode')
SRC_TEST_DIR = os.path.join(PROJECT_PATH, 'SRC-TestSuites')
GREEN_TEST_DIR = os.path.join(GREEN_CODE_DIR, 'GreenCode-TestSuites')

# Prompts
PROMPT_REFACTOR = """
Refactor this code to improve its efficiency, readability, and maintainability while keeping the functionality unchanged.
Ensure:
1. The refactored code is more efficient and optimized.
2. Add comments in the code where significant changes were made.

After the code, provide:
CHANGES_START
- [specific change description 1]
CHANGES_END

NEXT_STEPS_START
- [one concise recommendation for future improvement]
NEXT_STEPS_END
"""

PROMPT_GENERATE_TESTCASES = """
Create a comprehensive unit test case for the provided code.
Ensure:
1. The tests cover all edge cases and core functionality.

After the test code, provide:
CHANGES_START
- [test coverage description]
CHANGES_END

NEXT_STEPS_START
- [one concise recommendation for test improvement]
NEXT_STEPS_END
"""

# Initialize Groq client
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_client = Groq(api_key="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")

# Statistics tracking
processing_stats = {
    'start_time': time.time(),
    'total_files': 0,
    'total_loc': 0,
    'loc_by_type': {},
    'file_types': set(),
    'historical_totals': {'files': 0, 'loc': 0, 'time': 0.0}
}

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def remove_directory(path):
    """Delete directory and its contents."""
    if os.path.exists(path):
        shutil.rmtree(path)
        logging.info(f"Removed directory: {path}")

def should_exclude(file_path):
    """Check if file should be excluded from processing."""
    filename = os.path.basename(file_path)
    
    # Exclude specific files
    if filename in EXCLUDED_FILES:
        return True
    
    # Exclude specific directories
    excluded_dirs = ['SRC-TestSuites', 'GreenCode-TestSuites']
    if any(excluded_dir in file_path for excluded_dir in excluded_dirs):
        return True
    
    # Exclude test files (files containing 'Test' or 'test' in their names)
    if 'Test' in filename or 'test' in filename:
        return True
    
    return False

def clean_code_content(content):
    """Extract clean code from model response."""
    # Remove <think> blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Extract code from markdown blocks
    code_blocks = re.findall(r'```(?:python|java|javascript|typescript)?\n(.*?)```', content, re.DOTALL)
    
    if code_blocks:
        # Join all code blocks and strip whitespace
        return '\n\n'.join([block.strip() for block in code_blocks]).strip()
    
    # Fallback: Remove non-code lines
    return re.sub(r'^Here(.*?)\n', '', content, flags=re.MULTILINE).strip()

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda _: logging.warning("Retrying due to rate limit...")
)
def process_with_retry(prompt, content):
    """Process content with retry logic."""
    return groq_client.chat.completions.create(
        # model="deepseek-r1-distill-llama-70b",
        model="deepseek-r1-distill-qwen-32b",
        messages=[{"role": "user", "content": f"{prompt}\n\n{content}"}],
        temperature=0.1
    )

def process_file(file_path, prompt, output_dir, is_test=False):
    """Process file with Groq API and save results."""
    try:
        if should_exclude(file_path):
            logging.info(f"Skipping excluded file: {file_path}")
            return False

        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            original_loc = len(original_content.split('\n'))

        logging.info(f"Processing file: {os.path.basename(file_path)}")
        
        response = process_with_retry(prompt, original_content)
        response_content = response.choices[0].message.content
        
        # Clean and extract code
        cleaned_code = clean_code_content(response_content)
        
        # Extract changes and next steps
        changes = extract_section(response_content, "CHANGES_START", "CHANGES_END")
        next_steps = extract_section(response_content, "NEXT_STEPS_START", "NEXT_STEPS_END")

        # Prepare output path
        rel_path = os.path.relpath(file_path, PROJECT_PATH)
        output_path = os.path.join(output_dir, rel_path)
        
        if is_test:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}{TEST_SUFFIX}{ext}"

        ensure_directory(os.path.dirname(output_path))
        
        # Write cleaned code
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(cleaned_code)
        
        # Update statistics
        if not is_test:
            new_loc = len(cleaned_code.split('\n'))
            file_ext = os.path.splitext(file_path)[1]
            
            processing_stats['total_files'] += 1
            processing_stats['total_loc'] += new_loc
            processing_stats['file_types'].add(file_ext)
            
            if file_ext in processing_stats['loc_by_type']:
                processing_stats['loc_by_type'][file_ext] += new_loc
            else:
                processing_stats['loc_by_type'][file_ext] = new_loc
            
            log_to_csv(os.path.basename(file_path), changes, next_steps)
        
        return True

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return False

def extract_section(content, start_marker, end_marker):
    """Extract text between markers."""
    start_idx = content.find(start_marker)
    if start_idx == -1:
        return "No changes documented"
    
    start_idx += len(start_marker)
    end_idx = content.find(end_marker, start_idx)
    
    if end_idx == -1:
        extracted = content[start_idx:]
    else:
        extracted = content[start_idx:end_idx]
    
    # Clean extracted content
    return re.sub(r'^[\-\*]\s*', '', extracted.strip(), flags=re.MULTILINE)

def log_to_csv(file_name, changes, next_steps):
    """Log modifications to CSV with injection protection."""
    csv_path = os.path.join(PROJECT_PATH, "modification_overview.csv")
    headers = ["File Name", "Modification Timestamp", "Changes", "Next Steps"]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prevent CSV injection
    def sanitize(text):
        if text.startswith(('=', '-', '+', '@')):
            return f"'{text}"
        return text
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(headers)
            writer.writerow([
                sanitize(file_name),
                timestamp,
                sanitize(changes),
                sanitize(next_steps)
            ])
    except Exception as e:
        logging.error(f"CSV write failed: {str(e)}")

def generate_final_report():
    """Generate final overview report with historical data."""
    csv_path = os.path.join(PROJECT_PATH, "final_overview.csv")
    total_time = time.time() - processing_stats['start_time']
    
    # Read historical data if exists
    historical_data = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read and skip the header row
            for row in reader:
                if len(row) == 2:
                    key = row[0].strip()
                    value = row[1].strip()
                    # Skip section headers (e.g., "=== Last Run ===")
                    if key.startswith("===") or value.startswith("==="):
                        continue
                    # Convert numerical values
                    if 'LOC' in value:
                        historical_data[key] = int(value.replace(' LOC', ''))
                    elif 'Time' in key:
                        historical_data[key] = float(value)
                    elif value.isdigit():
                        historical_data[key] = int(value)
                    else:
                        historical_data[key] = value  # Keep as string if not numerical

    # Prepare last run data
    last_run_data = [
        ["Metric", "Value"],
        ["Total Files Modified (Last run)", processing_stats['total_files']],
        ["Total LOC Converted (Last run)", processing_stats['total_loc']],
        ["Total Time (minutes) (Last run)", round(total_time / 60, 2)]
    ]
    
    # Add file type breakdowns for last run
    for ext, loc in processing_stats['loc_by_type'].items():
        last_run_data.append([f"{ext} Files (last run)", f"{loc} LOC"])

    # Prepare historical data
    historical_totals = {
        "Total Files Modified": processing_stats['total_files'] + historical_data.get("Total Files Modified", 0),
        "Total LOC Converted": processing_stats['total_loc'] + historical_data.get("Total LOC Converted", 0),
        "Total Time (minutes)": round(total_time / 60, 2) + historical_data.get("Total Time (minutes)", 0)
    }

    # Add file type breakdowns for historical data
    for ext, loc in processing_stats['loc_by_type'].items():
        historical_key = f"{ext} Files"
        historical_totals[historical_key] = loc + historical_data.get(historical_key, 0)

    # Write to CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write Last Run Section
            writer.writerow(["=== Last Run ==="])
            writer.writerows(last_run_data)
            
            # Write Historical Section
            writer.writerow([])
            writer.writerow(["=== Historical Overview ==="])
            writer.writerow(["Metric", "Value"])
            for key, value in historical_totals.items():
                if 'LOC' in key:
                    writer.writerow([key, f"{value} LOC"])
                else:
                    writer.writerow([key, value])
                
        logging.info(f"Final report generated at: {csv_path}")
    except Exception as e:
        logging.error(f"Failed to generate final report: {str(e)}")

def main():
    # Setup directories
    remove_directory(GREEN_CODE_DIR)
    ensure_directory(GREEN_CODE_DIR)
    ensure_directory(SRC_TEST_DIR)
    ensure_directory(GREEN_TEST_DIR)

    try:
        # Step 1: Generate tests for original code
        logging.info("Starting test generation for original code...")
        for root, dirs, files in os.walk(PROJECT_PATH):
            if should_exclude(root):
                logging.info(f"Skipping directory: {root}")
                continue
            
            for file in files:
                if file.endswith(FILE_EXTENSIONS) and not should_exclude(file):
                    src_path = os.path.join(root, file)
                    logging.info(f"Processing file for test generation: {src_path}")
                    process_file(src_path, PROMPT_GENERATE_TESTCASES, SRC_TEST_DIR, is_test=True)

        # Step 2: Refine code
        logging.info("Starting code refinement...")
        for root, dirs, files in os.walk(PROJECT_PATH):
            if should_exclude(root):
                logging.info(f"Skipping directory: {root}")
                continue
            
            for file in files:
                if file.endswith(FILE_EXTENSIONS) and not should_exclude(file):
                    src_path = os.path.join(root, file)
                    logging.info(f"Processing file for refinement: {src_path}")
                    process_file(src_path, PROMPT_REFACTOR, GREEN_CODE_DIR)

        # Step 3: Generate tests for refined code
        logging.info("Starting test generation for refined code...")
        for root, dirs, files in os.walk(GREEN_CODE_DIR):
            # Skip the GreenCode-TestSuites directory
            if "GreenCode-TestSuites" in root:
                logging.info(f"Skipping directory: {root}")
                continue
            
            for file in files:
                if file.endswith(FILE_EXTENSIONS) and not should_exclude(file):
                    green_path = os.path.join(root, file)
                    logging.info(f"Processing refined file for test generation: {green_path}")
                    process_file(green_path, PROMPT_GENERATE_TESTCASES, GREEN_TEST_DIR, is_test=True)

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
    finally:
        # Generate final report
        generate_final_report()
        logging.info("Processing completed successfully")

if __name__ == "__main__":
    main()
