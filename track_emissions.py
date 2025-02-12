# Standard library imports
import os
import subprocess
import csv
import time
import logging
import shutil
from datetime import datetime

# Third-party library imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from codecarbon import EmissionsTracker
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots


# Load environment variables
env_path = os.path.abspath(".env")
load_dotenv(dotenv_path=env_path, verbose=True, override=True)
SOURCE_DIRECTORY = os.path.dirname(env_path)

GREEN_REFINED_DIRECTORY = os.path.join(SOURCE_DIRECTORY, "GreenCode")
RESULT_DIR = os.path.join(SOURCE_DIRECTORY, "Result")
REPORT_DIR = os.path.join(SOURCE_DIRECTORY, "Report")

# List of files and directories to exclude from processing
EXCLUDED_FILES = [
    file.strip() for file in os.getenv("EXCLUDED_FILES", "").split(",") if file.strip()
]
EXCLUDED_DIRECTORIES = [
    file.strip()
    for file in os.getenv("EXCLUDED_DIRECTORIES", "").split(",")
    if file.strip()
]

def is_test_file(file_path):
    """
    Check if a file is a test file based on its name and content.
    """
    file_name = os.path.basename(file_path).lower()
    return ('test' in file_name or 
            file_name.startswith('test_') or 
            file_name.endswith('_test.py') or
            file_name.endswith('test.java') or
            file_name.endswith('test.cpp') or
            file_name.endswith('test.cs'))

def count_lines_of_code(file_path, language="python"):
    """
    Count the number of lines of code in a file, excluding blank lines and comments,
    for multiple programming languages.
    
    Args:
        file_path (str): Path to the file to count lines.
        language (str): Programming language of the file (default: 'python').
    
    Returns:
        int: Number of lines of code.
    """
    # Define comment styles for supported languages
    comment_styles = {
        "python": {"single": "#", "multi_start": ('"""', "'''"), "multi_end": ('"""', "'''")},
        "javascript": {"single": "//", "multi_start": ("/*",), "multi_end": ("*/",)},
        "java": {"single": "//", "multi_start": ("/*",), "multi_end": ("*/",)},
        "c": {"single": "//", "multi_start": ("/*",), "multi_end": ("*/",)},
        "cpp": {"single": "//", "multi_start": ("/*",), "multi_end": ("*/",)},
    }
    
    if language not in comment_styles:
        raise ValueError(f"Unsupported language: {language}")
    
    styles = comment_styles[language]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        # Initialize counter
        loc = 0
        in_multiline_comment = False
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle multiline comments
            if in_multiline_comment:
                if any(line.endswith(end) for end in styles["multi_end"]):
                    in_multiline_comment = False
                continue
            
            if any(line.startswith(start) for start in styles["multi_start"]):
                if not any(line.endswith(end) for end in styles["multi_end"]):
                    in_multiline_comment = True
                continue
                
            # Skip single-line comments
            if line.startswith(styles["single"]):
                continue
            
            loc += 1
        
        return loc
    except Exception as e:
        logging.error(f"Error counting lines of code in {file_path}: {e}")
        return 0

def process_emissions_for_file(tracker, script_path, emissions_csv, file_type, result_dir, test_command):
    # First check if it's a test file
    if not is_test_file(script_path):
        return
    
    # If no test command, return immediately
    if not test_command:
        return
   
    emissions_data = None
    duration = 0
    test_output = 'Unknown'
    script_name = os.path.basename(script_path)

    # Count lines of code
    loc = count_lines_of_code(script_path)

    # Extract 'solution dir' (immediate parent directory)
    solution_dir = os.path.basename(os.path.dirname(script_path))
    is_green_refined = os.path.commonpath([script_path, GREEN_REFINED_DIRECTORY]) == GREEN_REFINED_DIRECTORY

    tracker_started = False
    try:
        # Start the emissions tracking ONLY for test files
        tracker = EmissionsTracker(allow_multiple_runs=True)
        tracker.start()
        tracker_started = True

        start_time = time.time()
        try:
            # Run test command for test files
            test_result = subprocess.run(test_command, capture_output=True, text=True, timeout=20)
            duration = time.time() - start_time
            test_output = 'Pass' if test_result.returncode == 0 else 'Fail'
        except subprocess.TimeoutExpired:
            test_output = 'Timeout'
    
    except Exception as e:
        logging.error(f"An error occurred while processing {script_name}: {e}")
        test_output = 'Error'

    finally:
        try:
            if tracker_started:
                emissions_data = tracker.stop()  # Stop the emissions tracking
        except Exception as e:
            logging.error(f"Error stopping the tracker for {script_name}: {e}")

    if emissions_data is not None:
        emissions_csv_default_path = 'emissions.csv'
        emissions_csv_target_path = os.path.join(result_dir, 'emissions.csv')
        try:
            if os.path.exists(emissions_csv_default_path):
                shutil.move(emissions_csv_default_path, emissions_csv_target_path)

            if os.stat(emissions_csv_target_path).st_size != 0:
                emissions_data = pd.read_csv(emissions_csv_target_path).iloc[-1]
                
                data = [
                    os.path.basename(script_path),
                    file_type,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{emissions_data['emissions'] * 1000:.6f}",
                    f"{duration:.2f}",
                    f"{emissions_data['emissions_rate'] * 1000:.6f}",
                    f"{emissions_data['cpu_power']:.6f}",
                    f"{emissions_data['gpu_power']:.6f}",
                    f"{emissions_data['ram_power']:.6f}",
                    f"{emissions_data['cpu_energy'] * 1000:.6f}",
                    f"{emissions_data['gpu_energy']:.6f}",
                    f"{emissions_data['ram_energy'] * 1000:.6f}",
                    f"{emissions_data['energy_consumed'] * 1000:.6f}",
                    test_output,
                    solution_dir,
                    is_green_refined,
                    loc  # Add the LOC count to the data
                ]
                with open(emissions_csv, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
                    file.flush()
            else:
                logging.error(f"No emissions data found for {script_path}")
        except Exception as e:
            logging.error(f"Error processing emissions data for {script_path}: {e}")
    else:
        logging.error(f"Emissions data collection failed for {script_name}")

def process_files_by_type(base_dir, emissions_data_csv, result_dir, file_extension, excluded_files, excluded_dirs, tracker, test_command_generator):
    files = []
    for root, dirs, file_list in os.walk(base_dir):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        # Additional check to ensure we're only processing files in the correct directory
        if base_dir == SOURCE_DIRECTORY:
            # For SOURCE_DIRECTORY, exclude files in GREEN_REFINED_DIRECTORY
            file_list = [f for f in file_list if GREEN_REFINED_DIRECTORY not in root]
        elif base_dir == GREEN_REFINED_DIRECTORY:
            # For GREEN_REFINED_DIRECTORY, only process files within this directory
            file_list = [f for f in file_list if GREEN_REFINED_DIRECTORY in root]

        for script in file_list:
            if (script.endswith(file_extension) and 
                script not in excluded_files and 
                is_test_file(os.path.join(root, script))):  # Only add test files
                script_path = os.path.join(root, script)
                test_command = test_command_generator(script_path)
                if test_command:
                    files.append((script_path, test_command))
    
    # Process test files
    for script_path, test_command in files:
        process_emissions_for_file(
            tracker=tracker,
            script_path=script_path,
            emissions_csv=emissions_data_csv,
            file_type=file_extension,
            result_dir=result_dir,
            test_command=test_command
        )
# Generate test commands for each language
def get_python_test_command(script_path):
    return [os.getenv('PYTEST_PATH', 'pytest'), script_path] if 'test' in script_path.lower() else None

# --------------- Improved version of get_java_test_command ----------------
def get_java_test_command(script_path):
    maven_path = os.getenv('MAVEN_PATH', 'mvn')
    test_name = os.path.splitext(os.path.basename(script_path))[0] + 'Test'
    return [maven_path, '-Dtest=' + test_name, 'test'] if 'test' in script_path.lower() else None
# ------------------------------------------------------------------------  

def get_cpp_test_command(script_path):
    if 'test' in script_path.lower():
        # Assuming a standard project structure
        test_file_name = os.path.basename(script_path).replace('.cpp', '_test.cpp')
        test_dir = os.path.join(os.path.dirname(script_path), 'test')
        test_file_path = os.path.join(test_dir, test_file_name)

        # Verify test file exists
        if not os.path.exists(test_file_path):
            logging.info(f"Warning: Test file {test_file_path} does not exist")
            return None
        
        # Create a temporary build directory
        build_dir = os.path.join(test_dir, 'build')
        os.makedirs(build_dir, exist_ok=True)
        
        # CMake command to configure the project
        cmake_path = os.getenv('GTEST_CMAKE_PATH', 'cmake')
        cmake_config_command = [
            cmake_path,
            f'-S{test_dir}',
            f'-B{build_dir}',
            '-DCMAKE_PREFIX_PATH=/usr/local',
            '-G', 'Unix Makefiles'
        ]
        
        # CMake build command
        cmake_build_command = [cmake_path, '--build', build_dir]
        
        # Test run command
        test_executable = os.path.join(build_dir, os.path.splitext(test_file_name)[0])        
        if not os.path.exists(test_executable):
            logging.warning(f"Test executable {test_executable} not found")
            return None
        
        return cmake_config_command + cmake_build_command + [test_executable]    
    return None

# ------------------ improved version of get_cs_test_command --------------
def get_cs_test_command(script_path):
    test_project = os.path.splitext(os.path.basename(script_path))[0]
    test_command = ['dotnet', 'test', test_project] if 'test' in script_path.lower() else None
    return test_command
# --------------------------------------------------------------------------


# Refactored process_folder function
def process_folder(base_dir, emissions_data_csv, result_dir, suffix, excluded_dirs):
    # Ensure the 'result' directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logging.info(f"Directory '{result_dir}' created successfully!")
    else:
        logging.info(f"Directory '{result_dir}' already exists.")
    
    # Check if the CSV file exists, if not, create it and write the header
    if not os.path.exists(emissions_data_csv):
        with open(emissions_data_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Application name", "File Type", "Timestamp", "Emissions (gCO2eq)",
                "Duration", "emissions_rate", "CPU Power (KWh)", "GPU Power (KWh)", "RAM Power (KWh)",
                "CPU Energy (Wh)", "GPU Energy (KWh)", "RAM Energy (Wh)", "Energy Consumed (Wh)", "Test Results", "solution dir", "Is Green Refined", "Lines of Code"
            ])
        logging.info(f"CSV file '{emissions_data_csv}' created with headers.")
    tracker = EmissionsTracker()

    # Process files for each language
    process_files_by_type(
        base_dir=base_dir,
        emissions_data_csv=emissions_data_csv,
        result_dir=result_dir,
        file_extension='.py',
        excluded_files=EXCLUDED_FILES,
        excluded_dirs=EXCLUDED_DIRECTORIES,
        tracker=tracker,
        test_command_generator=get_python_test_command
    )
    process_files_by_type(
        base_dir=base_dir,
        emissions_data_csv=emissions_data_csv,
        result_dir=result_dir,
        file_extension='.java',
        excluded_files=EXCLUDED_FILES,
        excluded_dirs=EXCLUDED_DIRECTORIES,
        tracker=tracker,
        test_command_generator=get_java_test_command
    )
    process_files_by_type(
        base_dir=base_dir,
        emissions_data_csv=emissions_data_csv,
        result_dir=result_dir,
        file_extension='.cpp',
        excluded_files=EXCLUDED_FILES,
        excluded_dirs=EXCLUDED_DIRECTORIES,
        tracker=tracker,
        test_command_generator=get_cpp_test_command
    )
    process_files_by_type(
        base_dir=base_dir,
        emissions_data_csv=emissions_data_csv,
        result_dir=result_dir,
        file_extension='.cs',
        excluded_files=EXCLUDED_FILES,
        excluded_dirs=EXCLUDED_DIRECTORIES,
        tracker=tracker,
        test_command_generator=get_cs_test_command
    )

    logging.info(f"Emissions data and test results written to {emissions_data_csv}")

# Call process_folder for 'before' and 'after' emissions data
process_folder(
    base_dir=SOURCE_DIRECTORY,
    emissions_data_csv=os.path.join(RESULT_DIR, 'main_before_emissions_data.csv'),
    result_dir=RESULT_DIR,
    suffix='before-in-detail',
    excluded_dirs=EXCLUDED_DIRECTORIES
)
process_folder(
    base_dir=GREEN_REFINED_DIRECTORY,
    emissions_data_csv=os.path.join(RESULT_DIR, 'main_after_emissions_data.csv'),
    result_dir=RESULT_DIR,
    suffix='after-in-detail',
    excluded_dirs=EXCLUDED_DIRECTORIES
)
logging.info("Emissions data processed successfully.")

# Compare emissions logic
def compare_emissions():
    # Load environment variables again (if needed)
    load_dotenv(dotenv_path=env_path, verbose=True, override=True)

    # Define paths to the before and after CSV files
    result_source_dir = os.path.join(SOURCE_DIRECTORY, 'Result', 'main_before_emissions_data.csv')
    result_green_refined_dir = os.path.join(SOURCE_DIRECTORY, 'Result', 'main_after_emissions_data.csv')

    # Check if both CSV files exist
    if not os.path.isfile(result_source_dir):
        logging.info(f"Source emissions data file not found: {result_source_dir}")
        return
    if not os.path.isfile(result_green_refined_dir):
        logging.info(f"Refined emissions data file not found: {result_green_refined_dir}")
        return

    # Read CSV files
    emissions_df = pd.read_csv(result_source_dir)
    emissions_after_df = pd.read_csv(result_green_refined_dir)

    # Merge dataframes on common columns
    try:
        merged_df = emissions_df.merge(
            emissions_after_df,
            on=["Application name", "File Type"],
            suffixes=('_before', '_after')
        )
    except KeyError as e:
        logging.info(f"Merge failed due to missing columns: {e}")
        return

    # Calculate the difference in emissions and determine the result
    merged_df['final emission'] = merged_df['Emissions (gCO2eq)_before'] - merged_df['Emissions (gCO2eq)_after']
    merged_df['Result'] = merged_df['final emission'].apply(lambda x: 'Improved' if x > 0 else 'Need improvement')

    # Select and rename columns
    result_df = merged_df[[
        "Application name",
        "File Type",
        "Timestamp_before",
        "Timestamp_after",
        "Emissions (gCO2eq)_before",
        "Emissions (gCO2eq)_after",
        "final emission",
        "Result"
    ]]
    result_df.columns = [
        "Application name",
        "File Type",
        "Timestamp (Before)",
        "Timestamp (After)",
        "Before",
        "After",
        "Final Emission",
        "Result"
    ]

    # Create 'Result' folder if it doesn't exist
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        logging.info(f"Directory '{RESULT_DIR}' created successfully!")
    else:
        logging.info(f"Directory '{RESULT_DIR}' already exists.")

    # Write to new CSV file
    result_file_path = os.path.join(RESULT_DIR, "comparison_results.csv")
    result_df.to_csv(result_file_path, index=False)

    logging.info(f"Comparison results saved to {result_file_path}")

# Call the compare_emissions function
compare_emissions()

def prepare_detailed_data(result_dir):
    comparison_csv = os.path.join(result_dir, "comparison_results.csv")
    before_csv = os.path.join(result_dir, 'main_before_emissions_data.csv')
    after_csv = os.path.join(result_dir, 'main_after_emissions_data.csv')
    
    # Read CSV files
    comparison_df = pd.read_csv(comparison_csv)
    before_df = pd.read_csv(before_csv)
    after_df = pd.read_csv(after_csv)
    
    # Merge before and after data
    merged_before = before_df[['Application name', 'File Type', 'Duration', 'Emissions (gCO2eq)', 'Energy Consumed (Wh)', 'solution dir']]
    merged_after = after_df[['Application name', 'File Type', 'Duration', 'Emissions (gCO2eq)', 'Energy Consumed (Wh)', 'solution dir']]
    
    # Group by 'solution dir'
    solution_dirs = sorted(comparison_df['Application name'].unique())  # Adjust as needed
    
    # Get unique solution directories
    solution_dirs = sorted(set(before_df['solution dir']).union(after_df['solution dir']))
    
    # Prepare data for each solution dir
    detailed_data = {}
    for dir in solution_dirs:
        before_details = merged_before[merged_before['solution dir'] == dir].to_dict(orient='records')
        after_details = merged_after[merged_after['solution dir'] == dir].to_dict(orient='records')
        detailed_data[dir] = {
            'before': before_details,
            'after': after_details
        }
    
    return solution_dirs, detailed_data

solution_dirs, detailed_data = prepare_detailed_data(RESULT_DIR)

def generate_html_report(result_dir, solution_dirs, detailed_data):
    # Initialize Jinja2 environment
    env = Environment(loader=FileSystemLoader(SOURCE_DIRECTORY))
    template_path = 'report_template.html'
    last_run_template_path = 'last_run_report_template.html'
    details_template_path = 'details_template.html'
    last_run_details_template_path = 'last_run_details_template.html'
    details_server_template_path = 'details_server_template.html'
    recommendations_template_path = 'recommendations_template.html'

    solution_dirs, detailed_data = prepare_detailed_data(result_dir)

    # Check if the templates exist
    for path in [details_template_path, template_path, last_run_details_template_path, 
                 last_run_template_path, details_server_template_path, recommendations_template_path]:
        if not os.path.isfile(os.path.join(SOURCE_DIRECTORY, path)):
            logging.error(f"Template file not found: {path}")
            return

    # Load the templates
    try:
        template = env.get_template(template_path)
        details_template = env.get_template(details_template_path)
        lastrun_template = env.get_template(last_run_template_path)
        lastrun_details_template = env.get_template(last_run_details_template_path)
        details_server_template = env.get_template(details_server_template_path)
        recommendations_template = env.get_template(recommendations_template_path)
    except Exception as e:
        logging.error(f"Failed to load templates: {e}")
        return

    # CSV paths
    csv_files = {
        'before': os.path.join(result_dir, 'main_before_emissions_data.csv'),
        'after': os.path.join(result_dir, 'main_after_emissions_data.csv'),
        'comparison': os.path.join(result_dir, 'comparison_results.csv'),
        'server': os.path.join(result_dir, 'server_data.csv'),
        'mul_server': os.path.join(result_dir, 'multiple_server_data.csv'),
        'recommendations': os.path.join(result_dir, 'modification_overview.csv'),
        'final_overview': os.path.join(result_dir, 'final_overview.csv')
    }

    # Load available CSVs
    dfs = {}
    for name, path in csv_files.items():
        if os.path.exists(path):
            try:
                dfs[name] = pd.read_csv(path)
                logging.info(f"Loaded CSV: {path}")
            except Exception as e:
                logging.error(f"Error reading {path}: {e}")
                dfs[name] = None
        else:
            logging.warning(f"CSV not found: {path}")
            dfs[name] = None

    # Extract DataFrames
    before_df = dfs.get('before')
    after_df = dfs.get('after')
    comparison_df = dfs.get('comparison')
    server_df = dfs.get('server')
    mul_server_df = dfs.get('mul_server')
    recommendations_df = dfs.get('recommendations')
    final_overview_df = dfs.get('final_overview')

    # Initialize default values for template variables
    template_vars = {
        'total_before': 0.0,
        'total_after': 0.0,
        'energy_table_html': "<p>Data unavailable</p>",
        'emissions_table_html': "<p>Data unavailable</p>",
        'total_emissions_before': 0.0,
        'total_emissions_after': 0.0,
        'last_run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'server_os_type_fig': "<p>OS type data unavailable</p>",
        'server_os_version_fig': "<p>OS version data unavailable</p>",
        'unique_hosts': 0,
        'average_co2_emission': 0.0,
        'average_energy_consumption': 0.0,
        'average_cpu_usage': 0.0,
        'average_ram_usage': 0.0,
        'average_disk_usage': 0.0,
        'average_network_usage': 0.0,
        'average_disk_read': 0.0,
        'average_disk_write': 0.0,
        'cpu_usage_data': [],
        'ram_usage_data': [],
        'disk_usage_data': [],
        'network_usage_data': [],
        'disk_read_data': [],
        'disk_write_data': [],
        'max_network': 0.0,
        'critical_servers': None,
        'div_combined_graph': "<p>Energy consumption data unavailable</p>",
        'div_emissions_combined_graph': "<p>Emissions data unavailable</p>",
        'div_pie_chart_non_embedded': "<p>Non-embedded code data unavailable</p>",
        'div_pie_chart_embedded': "<p>Embedded code data unavailable</p>",
        'unique_servers': [],
        'critical_server_count': 0,
        'total_last_run_co2': 0.0,
        'total_last_run_power': 0.0,
        'formatted_timestamp': "No data",
        'final_overview_data': None,
        'fresh_details': None,
        'total_files_modified_last_run': 0,
        'total_loc_converted_last_run': 0,
        'recommendations_details': None,
        'unique_dates': [],
        'server_details': []
    }

    # Process server data if available
    if server_df is not None:
        try:
            unique_hosts = server_df['IP address'].nunique()
            template_vars['unique_hosts'] = unique_hosts
            template_vars['average_co2_emission'] = round(server_df['CO2 emission (Metric Tons)'].sum() / unique_hosts, 2)
            template_vars['average_energy_consumption'] = round(server_df['Energy consumption (KWH)'].sum() / unique_hosts, 2)
            template_vars['average_cpu_usage'] = round(server_df['CPU usage (%)'].mean(), 2)
            template_vars['average_ram_usage'] = round(server_df['RAM usage (%)'].mean(), 2)
            template_vars['average_disk_usage'] = round(server_df['Disk usage (%)'].mean(), 2)
            template_vars['average_network_usage'] = round(server_df['Network usage (bytes)'].mean(), 2)
            template_vars['cpu_usage_data'] = server_df['CPU usage (%)'].tail(10).tolist()
            template_vars['ram_usage_data'] = server_df['RAM usage (%)'].tail(10).tolist()
            template_vars['disk_usage_data'] = server_df['Disk usage (%)'].tail(10).tolist()
            template_vars['network_usage_data'] = server_df['Network usage (bytes)'].tail(10).tolist()
        except KeyError as e:
            logging.error(f"Missing column in server data: {e}")

    # Process multi-server data if available
    if mul_server_df is not None:
        try:
            template_vars['average_disk_read'] = round(mul_server_df['disk_read_bytes'].mean(), 2)
            template_vars['average_disk_write'] = round(mul_server_df['disk_write_bytes'].mean(), 2)
            template_vars['disk_read_data'] = mul_server_df['disk_read_bytes'].tail(10).tolist()
            template_vars['disk_write_data'] = mul_server_df['disk_write_bytes'].tail(10).tolist()
            template_vars['max_network'] = max(template_vars['network_usage_data'])

            critical_servers = mul_server_df[mul_server_df['total_co2'] > 0.000001][['hostname', 'total_co2']]
            critical_servers = critical_servers.sort_values(by='total_co2', ascending=False).drop_duplicates(subset='hostname')
            template_vars['critical_server_count'] = len(critical_servers)
            template_vars['critical_servers'] = critical_servers.to_dict(orient='records') if not critical_servers.empty else None

            mul_server_df['timestamp'] = pd.to_datetime(mul_server_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            mul_server_df = mul_server_df.dropna(subset=['timestamp'])
            latest_timestamp = mul_server_df['timestamp'].max()
            last_run_records = mul_server_df[mul_server_df['timestamp'] == latest_timestamp]
            
            # Format the total_last_run_co2 to avoid scientific notation
            total_last_run_co2 = last_run_records['total_co2'].sum()
            total_last_run_power = last_run_records['total_power'].sum()
            template_vars['total_last_run_co2'] = f"{total_last_run_co2:.6f}"  # Format to 6 decimal places
            template_vars['total_last_run_power'] = f"{total_last_run_power:.6f}"  # Format to 6 decimal places
            # template_vars['total_last_run_power'] = last_run_records['total_power'].sum()
            if pd.notna(latest_timestamp):
                template_vars['formatted_timestamp'] = latest_timestamp.strftime('%d-%m-%Y %H:%M:%S')
            else:
                template_vars['formatted_timestamp'] = "N/A"  # Or any default value

            # Count of unique servers by 'os_type' and 'os_version'
            os_type_counts = mul_server_df.groupby(['os_type', 'hostname']).size().reset_index(name='count').groupby('os_type').agg({'hostname': 'count'}).rename(columns={'hostname': 'count'})
            os_version_counts = mul_server_df.groupby(['os_type', 'os_version', 'hostname']).size().reset_index(name='count').groupby(['os_type', 'os_version']).agg({'hostname': 'count'}).reset_index().rename(columns={'hostname': 'count'})
            color_palette = px.colors.qualitative.Pastel

            # Plot 1: Bar graph for 'os_type'
            fig1 = go.Figure(data=[go.Bar(x=os_type_counts.index, y=os_type_counts['count'], text=os_type_counts['count'], textposition='inside', textfont=dict(size=16), marker=dict(color=color_palette[:len(os_type_counts)]))])
            fig1.update_layout(title="Count of Unique Servers by OS Type", xaxis_title="OS Type", yaxis_title="Count of Unique Servers", template="plotly_white", width=600, height=300)
            template_vars['server_os_type_fig'] = pio.to_html(fig1, include_plotlyjs=False, full_html=False)

            # Plot 2: Bar graph for 'os_version' grouped by 'os_type'
            fig2 = go.Figure()
            for i, os_type in enumerate(os_version_counts['os_type'].unique()): os_data = os_version_counts[os_version_counts['os_type'] == os_type]; fig2.add_trace(go.Bar(x=os_data['os_version'], y=os_data['count'], name=os_type, text=os_data['count'], textposition='inside', textfont=dict(size=16), marker=dict(color=color_palette[i])))
            fig2.update_layout(title="Count of Unique Servers by OS Version (Grouped by OS Type)", xaxis_title="OS Version", yaxis_title="Count of Unique Servers", barmode='stack', template="plotly_white", width=600, height=400, xaxis_tickangle=-45, xaxis=dict(tickvals=os_version_counts['os_version'], ticktext=[os.replace(' ', '<br>') for os in os_version_counts['os_version']]), legend=dict(orientation="h", yanchor="top", y=1.20, xanchor="center", x=0.5, font=dict(size=12)), margin=dict(b=100))
            template_vars['server_os_version_fig'] = pio.to_html(fig2, include_plotlyjs=False, full_html=False)

            df = mul_server_df.copy()
            numeric_columns = ['cpu_percent', 'ram_total', 'ram_used', 'ram_percent', 'disk_read_bytes', 
                              'disk_write_bytes', 'total_power', 'cpu_power', 'ram_power', 'disk_base_power', 
                              'disk_io_power', 'total_co2', 'cpu_co2', 'ram_co2', 'disk_base_co2', 
                              'disk_io_co2', 'co2_factor', 'storage_device_count']
            string_columns = ['hostname', 'os_version', 'os_type', 'region', 'storage_devices', 'timestamp']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            for col in numeric_columns + string_columns:
                if col not in df.columns:
                    df[col] = 0 if col in numeric_columns else 'N/A'
            template_vars['server_details'] = df.to_dict(orient='records')
            template_vars['unique_servers'] = df['hostname'].unique().tolist()
        except KeyError as e:
            logging.error(f"Missing column in multi-server data: {e}")

    # Process recommendations data if available
    if recommendations_df is not None:
        try:
            recommendations_df['Modification Timestamp'] = pd.to_datetime(recommendations_df['Modification Timestamp'], errors='coerce')
            recommendations_df = recommendations_df.dropna(subset=['Modification Timestamp'])
            unique_dates = recommendations_df['Modification Timestamp'].dt.date.unique().tolist()
            recommendations_details = {}
            for date in unique_dates:
                date_records = recommendations_df[recommendations_df['Modification Timestamp'].dt.date == date]
                recommendations_details[str(date)] = date_records.assign(
                    **{"Modification Timestamp": date_records['Modification Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')}
                ).to_dict(orient='records')
            template_vars['recommendations_details'] = recommendations_details
            template_vars['unique_dates'] = unique_dates
        except KeyError as e:
            logging.error(f"Missing column in recommendations data: {e}")

    # Process final_overview data if available
    if final_overview_df is not None:
        try:
            # Extract data from final_overview.csv
            final_overview_data = {
                'fresh_details': {
                    'total_files_modified_last_run': final_overview_df.loc[
                        final_overview_df['Metric'] == 'Total Files Modified (Last run)', 'Value'].values[0],
                    'total_loc_converted_last_run': final_overview_df.loc[
                        final_overview_df['Metric'] == 'Total LOC Converted (Last run)', 'Value'].values[0],
                    'total_time_minutes_last_run': final_overview_df.loc[
                        final_overview_df['Metric'] == 'Total Time (minutes) (Last run)', 'Value'].values[0],
                },
                'historical_overview': {
                    'total_files_modified': final_overview_df.loc[
                        final_overview_df['Metric'] == 'Total Files Modified', 'Value'].values[0],
                    'total_loc_converted': final_overview_df.loc[
                        final_overview_df['Metric'] == 'Total LOC Converted', 'Value'].values[0],
                    'total_time_minutes': final_overview_df.loc[
                        final_overview_df['Metric'] == 'Total Time (minutes)', 'Value'].values[0],
                }
            }
            
            # Dynamically add file type information
            file_type_last_run = final_overview_df[final_overview_df['Metric'].str.contains(r'\.\w+ Files \(Last run\)')]
            if not file_type_last_run.empty:
                final_overview_data['fresh_details']['file_types_last_run'] = {
                    row['Metric'].split()[0]: row['Value'] 
                    for _, row in file_type_last_run.iterrows()
                }
            
            file_types = final_overview_df[final_overview_df['Metric'].str.contains(r'\.\w+ Files$')]
            if not file_types.empty:
                final_overview_data['historical_overview']['file_types'] = {
                    row['Metric'].split()[0]: row['Value'] 
                    for _, row in file_types.iterrows()
                }
            
            template_vars['final_overview_data'] = final_overview_data
        except Exception as e:
            logging.error(f"Error processing final_overview.csv: {e}")
            # Set default values if final_overview.csv is missing or has incorrect data
            template_vars['final_overview_data'] = {
                'fresh_details': {
                    'total_files_modified_last_run': 0,
                    'total_loc_converted_last_run': 0,
                    'total_time_minutes_last_run': 0,
                    'file_types_last_run': {}
                },
                'historical_overview': {
                    'total_files_modified': 0,
                    'total_loc_converted': 0,
                    'total_time_minutes': 0,
                    'file_types': {}
                }
            }
    else:
        logging.warning("final_overview.csv not found or could not be loaded.")
        # Set default values if final_overview.csv is missing or could not be loaded
        template_vars['final_overview_data'] = {
            'fresh_details': {
                'total_files_modified_last_run': 0,
                'total_loc_converted_last_run': 0,
                'total_time_minutes_last_run': 0,
                'file_types_last_run': {}
            },
            'historical_overview': {
                'total_files_modified': 0,
                'total_loc_converted': 0,
                'total_time_minutes': 0,
                'file_types': {}
            }
        }

    # Process before and after data if available
    if before_df is not None and after_df is not None:
        try:
            def is_test_application(app_name):
                return 'test' in str(app_name).lower()

            template_vars['total_before'] = before_df[before_df['Application name'].apply(is_test_application)]['Energy Consumed (Wh)'].astype(float).sum()
            template_vars['total_after'] = after_df[after_df['Application name'].apply(is_test_application)]['Energy Consumed (Wh)'].astype(float).sum()

            if not before_df.empty:
                latest_before_df = before_df.loc[[before_df['Timestamp'].idxmax()]]
                template_vars['latest_before_details'] = [latest_before_df[['Application name', 'File Type', 'Duration', 'Emissions (gCO2eq)', 'Energy Consumed (Wh)', 'solution dir']].to_dict()]
                template_vars['latest_total_before'] = latest_before_df[latest_before_df['Application name'].apply(is_test_application)]['Energy Consumed (Wh)'].astype(float).sum()
            else:
                template_vars['latest_before_details'] = []
                template_vars['latest_total_before'] = 0.0

            if not after_df.empty:
                latest_after_df = after_df.loc[[after_df['Timestamp'].idxmax()]]
                template_vars['latest_after_details'] = [latest_after_df[['Application name', 'File Type', 'Duration', 'Emissions (gCO2eq)', 'Energy Consumed (Wh)', 'solution dir']].to_dict()]
                template_vars['latest_total_after'] = latest_after_df[latest_after_df['Application name'].apply(is_test_application)]['Energy Consumed (Wh)'].astype(float).sum()
            else:
                template_vars['latest_after_details'] = []
                template_vars['latest_total_after'] = 0.0

            template_vars['before_details'] = before_df[['Application name', 'File Type', 'Duration', 'Emissions (gCO2eq)', 'Energy Consumed (Wh)', 'solution dir']].to_dict(orient='records')
            template_vars['after_details'] = after_df[['Application name', 'File Type', 'Duration', 'Emissions (gCO2eq)', 'Energy Consumed (Wh)', 'solution dir']].to_dict(orient='records')

            if comparison_df is not None:
                template_vars['total_emissions_before'] = comparison_df[comparison_df['Application name'].apply(is_test_application)]['Before'].astype(float).sum()
                template_vars['total_emissions_after'] = comparison_df[comparison_df['Application name'].apply(is_test_application)]['After'].astype(float).sum()
                template_vars['latest_total_emissions_before'] = latest_before_df[latest_before_df['Application name'].apply(is_test_application)]['Emissions (gCO2eq)'].astype(float).sum()
                template_vars['latest_total_emissions_after'] = latest_after_df[latest_after_df['Application name'].apply(is_test_application)]['Emissions (gCO2eq)'].astype(float).sum()

            # Generate energy consumption graph
            before_file_type = before_df.groupby('solution dir')['Energy Consumed (Wh)'].sum().reset_index()
            after_file_type = after_df.groupby('solution dir')['Energy Consumed (Wh)'].sum().reset_index()
            before_file_type_sorted = before_file_type.sort_values('Energy Consumed (Wh)', ascending=False)
            after_file_type_sorted = after_file_type.sort_values('Energy Consumed (Wh)', ascending=False)
            unique_solution_dirs = sorted(set(before_file_type_sorted['solution dir']).union(after_file_type_sorted['solution dir']))
            colors = px.colors.qualitative.Pastel
            color_mapping = {solution_dir: colors[i % len(colors)] for i, solution_dir in enumerate(before_file_type_sorted['solution dir'].unique())}

            fig = go.Figure()
            for i, (_, row) in enumerate(before_file_type_sorted.iterrows()):
                fig.add_trace(go.Bar(x=[row['solution dir']], y=[row['Energy Consumed (Wh)']], name='Before', 
                                     marker=dict(color=color_mapping.get(row['solution dir'], 'grey'), opacity=0.9), 
                                     offsetgroup=0, showlegend=True if i == 0 else False))
            for i, (_, row) in enumerate(after_file_type_sorted.iterrows()):
                fig.add_trace(go.Bar(x=[row['solution dir']], y=[row['Energy Consumed (Wh)']], name='After', 
                                     marker=dict(color=color_mapping.get(row['solution dir'], 'grey'), opacity=0.6, 
                                                 pattern=dict(shape="/", solidity=0.7)), offsetgroup=1, 
                                     showlegend=True if i == 0 else False))
            fig.update_layout(title={'text': 'Source Code Directory Level Energy Consumption (Wh) - Before vs After Optimization', 
                                    'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, 
                             xaxis_title='Solution Dir', yaxis_title='Energy Consumed (Wh)', barmode='group', 
                             xaxis=dict(tickangle=45, tickformat=".6f"), yaxis=dict(range=[0, max(
                                 before_file_type_sorted['Energy Consumed (Wh)'].max(), 
                                 after_file_type_sorted['Energy Consumed (Wh)'].max()) * 1.1], tickformat=".6f"), 
                             margin=dict(l=50, r=50, t=100, b=120), showlegend=False, width=700, height=400, 
                             plot_bgcolor='white', paper_bgcolor='white')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            template_vars['div_combined_graph'] = pio.to_html(fig, include_plotlyjs=False, full_html=False)

            # Generate emissions graph
            before_gco2eq = before_df.groupby('solution dir')['Emissions (gCO2eq)'].sum().reset_index()
            after_gco2eq = after_df.groupby('solution dir')['Emissions (gCO2eq)'].sum().reset_index()
            before_gco2eq_sorted = before_gco2eq.sort_values('Emissions (gCO2eq)', ascending=False)
            after_gco2eq_sorted = after_gco2eq.sort_values('Emissions (gCO2eq)', ascending=False)

            fig = go.Figure()
            for i, (_, row) in enumerate(before_gco2eq_sorted.iterrows()):
                fig.add_trace(go.Bar(x=[row['solution dir']], y=[row['Emissions (gCO2eq)']], name='Before',
                                     marker=dict(color=color_mapping.get(row['solution dir'], 'grey'), opacity=0.9),
                                     offsetgroup=0, showlegend=True if i == 0 else False))
            for i, (_, row) in enumerate(after_gco2eq_sorted.iterrows()):
                fig.add_trace(go.Bar(x=[row['solution dir']], y=[row['Emissions (gCO2eq)']], name='After',
                                     marker=dict(color=color_mapping.get(row['solution dir'], 'grey'), opacity=0.6,
                                                 pattern=dict(shape="/", solidity=0.7)), offsetgroup=1,
                                     showlegend=True if i == 0 else False))
            fig.update_layout(title={'text': 'Source Code Directory Level Emissions (gCO2eq) - Before vs After Optimization',
                                    'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                             xaxis_title='Solution Dir', yaxis_title='Emissions (gCO2eq)', barmode='group',
                             xaxis=dict(tickangle=45, tickformat=".6f"), yaxis=dict(range=[0, max(
                                 before_gco2eq_sorted['Emissions (gCO2eq)'].max(),
                                 after_gco2eq_sorted['Emissions (gCO2eq)'].max()) * 1.1], tickformat=".6f"),
                             margin=dict(l=50, r=50, t=100, b=120), showlegend=False, width=700, height=400,
                             plot_bgcolor='white', paper_bgcolor='white')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            template_vars['div_emissions_combined_graph'] = pio.to_html(fig, include_plotlyjs=False, full_html=False)

            # Generate top five tables
            top_five_energy_before = before_df.sort_values('Energy Consumed (Wh)', ascending=False).head(5)[
                ['Application name', 'Timestamp', 'Energy Consumed (Wh)']]
            top_five_energy_before.rename(columns={'Application name': 'File Name', 'Timestamp': 'Timestamp',
                                                  'Energy Consumed (Wh)': 'Energy Consumed (Wh)'}, inplace=True)
            template_vars['energy_table_html'] = top_five_energy_before.to_html(index=False, classes='table', border=0,
                                                                               float_format=lambda x: f"{x:.6f}")

            top_five_emissions_before = before_df.sort_values('Emissions (gCO2eq)', ascending=False).head(5)[
                ['Application name', 'Timestamp', 'Emissions (gCO2eq)']]
            top_five_emissions_before.rename(columns={'Application name': 'File Name', 'Timestamp': 'Timestamp',
                                                     'Emissions (gCO2eq)': 'Emissions (gCO2eq)'}, inplace=True)
            template_vars['emissions_table_html'] = top_five_emissions_before.to_html(index=False, classes='table',
                                                                                     border=0, float_format=lambda x: f"{x:.6f}")

            # Generate pie charts for embedded and non-embedded code
            embedded_types = ['.html', '.css', '.xml', '.php', '.ts']
            non_embedded_types = ['.py', '.java', '.cpp', '.rb']
            embedded_df = comparison_df[comparison_df['File Type'].isin(embedded_types)]
            non_embedded_df = comparison_df[comparison_df['File Type'].isin(non_embedded_types)]

            if not embedded_df.empty:
                total_embedded_before = embedded_df['Before'].astype(float).sum()
                total_embedded_after = embedded_df['After'].astype(float).sum()
                reduction_percentage = ((total_embedded_before - total_embedded_after) / total_embedded_before * 100)
                fig = go.Figure(data=[go.Pie(values=[total_embedded_before, total_embedded_after],
                                             labels=['Before', 'After'], marker=dict(colors=['#FF6B6B', '#4ECDC4']),
                                             textinfo='label+value', textposition='outside',
                                             texttemplate='%{label}<br>%{value:.6f} gCO2eq', hole=0.7,
                                             direction='clockwise', showlegend=False)])
                fig.update_layout(title=dict(text='Embedded Code Emissions (gCO2eq)<br>[".html",".css",".xml",".php",".ts"]',
                                             y=0.95, x=0.5, xanchor='center', yanchor='top', font=dict(size=18)),
                                 annotations=[dict(text=f"↓{reduction_percentage:.1f}%", x=0.5, y=0.5,
                                                   font=dict(size=24, color='green'), showarrow=False),
                                             dict(text="Reduction", x=0.5, y=0.42, font=dict(size=14, color='green'),
                                                  showarrow=False)], width=600, height=400, paper_bgcolor='white',
                                 plot_bgcolor='white', showlegend=False)
                template_vars['div_pie_chart_embedded'] = pio.to_html(fig, include_plotlyjs=False, full_html=False)

            if not non_embedded_df.empty:
                total_non_embedded_before = non_embedded_df['Before'].astype(float).sum()
                total_non_embedded_after = non_embedded_df['After'].astype(float).sum()
                reduction_percentage = ((total_non_embedded_before - total_non_embedded_after) / total_non_embedded_before * 100)
                fig = go.Figure(data=[go.Pie(values=[total_non_embedded_before, total_non_embedded_after],
                                             labels=['Before', 'After'], marker=dict(colors=['#FF6B6B', '#4ECDC4']),
                                             textinfo='label+value', textposition='outside',
                                             texttemplate='%{label}<br>%{value:.6f} gCO2eq', hole=0.7,
                                             direction='clockwise', showlegend=False)])
                fig.update_layout(title=dict(text='Non-Embedded Code Emissions (gCO2eq)<br>[".py", ".java", ".cpp", ".rb"]',
                                             y=0.95, x=0.5, xanchor='center', yanchor='top', font=dict(size=18)),
                                 annotations=[dict(text=f"↓{reduction_percentage:.1f}%", x=0.5, y=0.5,
                                                   font=dict(size=24, color='green'), showarrow=False),
                                             dict(text="Reduction", x=0.5, y=0.38, font=dict(size=14, color='green'),
                                                  showarrow=False)], width=600, height=400, paper_bgcolor='white',
                                 plot_bgcolor='white', showlegend=False)
                template_vars['div_pie_chart_non_embedded'] = pio.to_html(fig, include_plotlyjs=False, full_html=False)

        except Exception as e:
            logging.error(f"Error processing before/after data: {e}")

    # Render the templates with dynamic data
    try:
        html_content = template.render(**template_vars)
        html_details_content = details_template.render(
            solution_dirs=solution_dirs,  # Directly use the parameter
            before_details=template_vars.get('before_details', []),
            after_details=template_vars.get('after_details', []),
            detailed_data=detailed_data  # Directly use the parameter
        )
        timestamp_html_content = lastrun_template.render(
            latest_total_before=f"{template_vars.get('latest_total_before', 0.0):.2f}",
            latest_total_after=f"{template_vars.get('latest_total_after', 0.0):.2f}",
            latest_energy_table_html=template_vars.get('latest_energy_table_html', "<p>Data unavailable</p>"),
            latest_emissions_table_html=template_vars.get('latest_emissions_table_html', "<p>Data unavailable</p>"),
            latest_total_emissions_before=f"{template_vars.get('latest_total_emissions_before', 0.0):.2f}",
            latest_total_emissions_after=f"{template_vars.get('latest_total_emissions_after', 0.0):.2f}",
            last_run_timestamp=template_vars.get('last_run_timestamp', "No data"),
            unique_hosts=template_vars.get('unique_hosts', 0),
            average_co2_emission=round(template_vars.get('average_co2_emission', 0.0), 4),
            average_energy_consumption=round(template_vars.get('average_energy_consumption', 0.0), 4),
            average_cpu_usage=round(template_vars.get('average_cpu_usage', 0.0), 2),
            average_ram_usage=round(template_vars.get('average_ram_usage', 0.0), 2),
            average_disk_usage=round(template_vars.get('average_disk_usage', 0.0), 2),
            average_network_usage=round(template_vars.get('average_network_usage', 0.0), 2),
            cpu_usage_data=template_vars.get('cpu_usage_data', []),
            ram_usage_data=template_vars.get('ram_usage_data', []),
            disk_usage_data=template_vars.get('disk_usage_data', []),
            network_usage_data=template_vars.get('network_usage_data', []),
            disk_read_data=template_vars.get('disk_read_data', []),
            disk_write_data=template_vars.get('disk_write_data', []),
            max_network=round(template_vars.get('max_network', 0.0), 2),
            critical_servers=template_vars.get('critical_servers', None),
            div_combined_graph=template_vars.get('div_combined_graph', "<p>Energy consumption data unavailable</p>"),
            div_emissions_combined_graph=template_vars.get('div_emissions_combined_graph', "<p>Emissions data unavailable</p>"),
            div_pie_chart_non_embedded=template_vars.get('div_pie_chart_non_embedded', "<p>Non-embedded code data unavailable</p>"),
            div_pie_chart_embedded=template_vars.get('div_pie_chart_embedded', "<p>Embedded code data unavailable</p>"),
            final_overview_data=template_vars.get('final_overview_data', {})
        )
        timestamp_html_details_content = lastrun_details_template.render(
            solution_dirs=solution_dirs,  # Directly use the parameter
            latest_before_details=template_vars.get('latest_before_details', []),
            latest_after_details=template_vars.get('latest_after_details', []),
            detailed_data=detailed_data  # Directly use the parameter
        )
        server_details = details_server_template.render(
            unique_servers=template_vars.get('unique_servers', []),
            server_details=template_vars.get('server_details', [])
        )
        recommendations_detalis = recommendations_template.render(
            unique_dates=template_vars.get('unique_dates', []),
            recommendations_details=template_vars.get('recommendations_details', {}),
            final_overview_data=template_vars.get('final_overview_data', {})
        )
    except Exception as e:
        logging.error(f"Template rendering failed: {e}")
        return

    # Save the reports
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M')
    date_folder_path = os.path.join(REPORT_DIR, current_date)
    time_folder_path = os.path.join(date_folder_path, current_time)
    os.makedirs(time_folder_path, exist_ok=True)

    report_paths = {
        'details_report': os.path.join(time_folder_path, 'details_report.html'),
        'emissions_report': os.path.join(time_folder_path, 'emissions_report.html'),
        'server_report': os.path.join(time_folder_path, 'server_report.html'),
        'recommendations_report': os.path.join(time_folder_path, 'recommendations_report.html'),
        'main_report': os.path.join(REPORT_DIR, 'emissions_report.html'),
        'detailed_report': os.path.join(REPORT_DIR, 'details_report.html'),
        'server_main_report': os.path.join(REPORT_DIR, 'server_report.html'),
        'recommendations_main_report': os.path.join(REPORT_DIR, 'recommendations_report.html')
    }

    for name, path in report_paths.items():
        try:
            with open(path, 'w', encoding="utf-8") as f:
                if name == 'details_report':
                    f.write(timestamp_html_details_content)
                elif name == 'emissions_report':
                    f.write(timestamp_html_content)
                elif name == 'server_report':
                    f.write(server_details)
                elif name == 'recommendations_report':
                    f.write(recommendations_detalis)
                elif name == 'main_report':
                    f.write(html_content)
                elif name == 'detailed_report':
                    f.write(html_details_content)
                elif name == 'server_main_report':
                    f.write(server_details)
                elif name == 'recommendations_main_report':
                    f.write(recommendations_detalis)
            logging.info(f"Report generated at {path}")
        except Exception as e:
            logging.error(f"Failed to save report {name}: {e}")

# Generate HTML report
generate_html_report(RESULT_DIR, solution_dirs=solution_dirs, detailed_data=detailed_data)
