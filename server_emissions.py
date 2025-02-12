import os
import json
import dotenv
import uuid
import time
import logging
import pandas as pd
import psutil
import socket
from datetime import datetime
from dotenv import load_dotenv
import sqlite3

# Load environment variables
env_path = os.path.abspath(".env")
load_dotenv(dotenv_path=env_path, verbose=True, override=True)

# Setup Logging
LOG_FILE = os.path.join(os.path.dirname(env_path), 'server_emissions.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Directory and file path constants
BYTES_TO_GB = 1024 ** 3
MILLI_WATTS_TO_KILOWATTS = 1000
DEFAULT_SLEEP_TIME = int(os.getenv('DEFAULT_SLEEP_TIME', 20))
RUN_TIME_IN_MINUTES = int(os.getenv('RUN_TIME_IN_MINUTES', 1))
# RESULT_DIR = os.path.join(BASE_DIR, 'GreenCode')
RESULT_DIR = os.path.join(os.path.dirname(env_path), 'Result')
CSV_FILE = os.path.join(RESULT_DIR, 'server_data.csv')


# Load CO2 emission factors from .env with defaults
GLOBAL_GRID_CO2_FACTOR = float(os.getenv('GLOBAL_GRID_CO2_FACTOR', 0.54))
US_GRID_CO2_FACTOR = float(os.getenv('US_GRID_CO2_FACTOR', 0.46))
GLOBAL_RENEWABLE_CO2_FACTOR = float(os.getenv('GLOBAL_RENEWABLE_CO2_FACTOR', 0.01))

def ensure_result_directory_exists():
    """Ensure that the Result directory exists."""
    try:
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
            logger.info(f"Directory '{RESULT_DIR}' created successfully!")
        else:
            logger.info(f"Directory '{RESULT_DIR}' already exists.")
    except Exception as e:
        logger.error(f"Failed to create directory '{RESULT_DIR}': {e}")
        raise

def get_system_info(previous_network):
    """
    Gather system information and calculate energy consumption and CO2 emission.

    Args:
        previous_network (tuple): Previous (bytes_sent, bytes_recv).

    Returns:
        dict: Collected system metrics.
        tuple: Current (bytes_sent, bytes_recv).
    """
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent

        # Calculate network usage delta
        current_network = psutil.net_io_counters()
        bytes_sent = current_network.bytes_sent
        bytes_recv = current_network.bytes_recv
        network_delta = (bytes_sent - previous_network[0], bytes_recv - previous_network[1])
        total_network_usage = network_delta[0] + network_delta[1]

        energy_consumption = get_energy_consumption(cpu_usage, ram_usage, disk_usage, total_network_usage)
        co2_emission = calculate_co2_emission(energy_consumption)

        system_info = {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Host-name': hostname,
            'IP address': ip_address,
            'CPU usage (%)': cpu_usage,
            'RAM usage (%)': ram_usage,
            'Disk usage (%)': disk_usage,
            'Network usage (bytes)': total_network_usage,
            'Energy consumption (KWH)': round(energy_consumption, 4),
            'CO2 emission (Metric Tons)': round(co2_emission * 1e6, 2)
            # 'CO2 emission (Kilograms)': round(co2_emission * 1e3, 2)  # Scale up to Kilograms
            # 'CO2 emission (kt)': round(co2_emission, 4)
        }

        return system_info, (bytes_sent, bytes_recv)

    except Exception as e:
        logger.error(f"Error gathering system info: {e}")
        raise

def get_max_power_consumption():
    """Calculate the maximum power consumption based on system specs."""
    try:
        cpu_max_freq = psutil.cpu_freq().max  # in MHz
        ram_size_gb = psutil.virtual_memory().total / BYTES_TO_GB  # in GB
        disk_size_gb = sum(psutil.disk_usage(part.mountpoint).total for part in psutil.disk_partitions()) / BYTES_TO_GB  # in GB
        max_power_consumption = calculate_power_consumption(cpu_max_freq, ram_size_gb, disk_size_gb)
        return max_power_consumption
    except Exception as e:
        logger.error(f"Error calculating max power consumption: {e}")
        raise

def calculate_power_consumption(cpu_max_freq, ram_size_gb, disk_size_gb):
    """
    Calculate power consumption based on CPU frequency, RAM size, and disk size.

    Args:
        cpu_max_freq (float): CPU max frequency in MHz.
        ram_size_gb (float): RAM size in GB.
        disk_size_gb (float): Disk size in GB.

    Returns:
        float: Maximum power consumption in watts.
    """
    try:
        CPU_FREQ_FACTOR = 1000
        RAM_SIZE_FACTOR = 16
        DISK_SIZE_FACTOR = 1000
        BASE_POWER_CONSUMPTION = 200  # in watts
        LOAD_FACTOR = 1.0

        cpu_factor = cpu_max_freq / CPU_FREQ_FACTOR
        ram_factor = ram_size_gb / RAM_SIZE_FACTOR
        disk_factor = disk_size_gb / DISK_SIZE_FACTOR

        total_factor = cpu_factor + ram_factor + disk_factor + LOAD_FACTOR
        max_power_consumption = total_factor * BASE_POWER_CONSUMPTION
        return max_power_consumption
    except Exception as e:
        logger.error(f"Error calculating power consumption: {e}")
        raise

def get_energy_consumption(cpu_usage, ram_usage, disk_usage, network_usage):
    """
    Calculate energy consumption based on system usage metrics.

    Args:
        cpu_usage (float): CPU usage percentage.
        ram_usage (float): RAM usage percentage.
        disk_usage (float): Disk usage percentage.
        network_usage (int): Network usage in bytes.

    Returns:
        float: Energy consumption in kilowatt-hours (KWH).
    """
    try:
        max_power_consumption = get_max_power_consumption()

        cpu_factor = cpu_usage / 100
        ram_factor = ram_usage / 100
        disk_factor = disk_usage / 100
        network_factor = network_usage / (BYTES_TO_GB / MILLI_WATTS_TO_KILOWATTS)  # Convert bytes to kilowatts

        total_factor = (cpu_factor + ram_factor + disk_factor + network_factor) / 4
        energy_consumption = (total_factor * max_power_consumption) / MILLI_WATTS_TO_KILOWATTS  # Convert to KWH

        return energy_consumption
    except Exception as e:
        logger.error(f"Error calculating energy consumption: {e}")
        raise

def calculate_co2_emission(energy_consumption):
    try:
        co2_emission_factors = {
            'grid': {
                'global': GLOBAL_GRID_CO2_FACTOR,
                'us': US_GRID_CO2_FACTOR,
            },
            'renewable': {
                'global': GLOBAL_RENEWABLE_CO2_FACTOR,
            },
        }
        energy_source = os.getenv('ENERGY_SOURCE', 'grid')  # 'grid' or 'renewable'
        location = os.getenv('LOCATION', 'global')  # 'global' or 'us'

        co2_emission_per_kwh = co2_emission_factors.get(energy_source, {}).get(location, GLOBAL_GRID_CO2_FACTOR)
        co2_emission = (energy_consumption * co2_emission_per_kwh) / MILLI_WATTS_TO_KILOWATTS  # Convert to kt

        return co2_emission
    except Exception as e:
        logger.error(f"Error calculating CO2 emission: {e}")
        raise

def update_csv(system_info):
    """
    Append system information to the CSV file.

    Args:
        system_info (dict): Dictionary containing system metrics.
    """
    try:
        ensure_result_directory_exists()
        # If CSV doesn't exist, create it with headers
        if not os.path.exists(CSV_FILE):
            df = pd.DataFrame(columns=system_info.keys())
            df.to_csv(CSV_FILE, index=False)
            logger.info(f"CSV file '{CSV_FILE}' created with headers.")

        # Append the new data
        df_new = pd.DataFrame([system_info])
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
        logger.info(f"Appended new data to '{CSV_FILE}'.")
    except Exception as e:
        logger.error(f"Error updating CSV file '{CSV_FILE}': {e}")
        raise

def main():
    """Main function to collect and log system emissions data."""
    try:
        previous_network = (0, 0)
        # Initialize previous_network with current network stats
        current_network = psutil.net_io_counters()
        previous_network = (current_network.bytes_sent, current_network.bytes_recv)

        start_time = time.time()
        end_time = start_time + (RUN_TIME_IN_MINUTES * 60)

        while time.time() < end_time:
            # remaining_time = int(end_time - time.time())
            # logging.info(f"\rTime remaining: {remaining_time} seconds", end="")  # logging.info dynamic countdown
            system_info, previous_network = get_system_info(previous_network)
            update_csv(system_info)
            logger.info(f"Collected and logged data: {system_info}")
            time.sleep(DEFAULT_SLEEP_TIME)

        logger.info("Data collection completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()
