import pandas as pd
import paramiko
import concurrent.futures
import time
from datetime import datetime
import os
from threading import Lock
import logging
import wmi
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.abspath(".env")
load_dotenv(dotenv_path=env_path, verbose=True, override=True)

# Define the result directory
RESULT_DIR = os.path.join(os.path.dirname(env_path), 'Result')

# Ensure the result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)

class RemoteSystemMonitor:
    def __init__(self, measurement_interval=1,
                 pue=1.67,  # Typical PUE value
                 cpu_idle_power=10,  # Watts
                 cpu_max_power=100,   # Watts
                 ram_idle_power_per_gb=0.3,    # Watts/GB
                 ram_active_power_per_gb=0.6,  # Watts/GB
                 disk_power=8,       # Watts per disk
                 k_read=8e-8,    # Joules/byte
                 k_write=8e-8,   # Joules/byte
                 # CO2 emission factors (kg CO2 per kWh) for different regions
                 co2_factors={
                     'global': 0.475,      # Global average
                     'us': 0.385,          # United States
                     'eu': 0.231,          # European Union
                     'china': 0.555,       # China
                     'india': 0.708,       # India
                     'uk': 0.233,          # United Kingdom
                     'japan': 0.474,       # Japan
                     'australia': 0.656,    # Australia
                     'canada': 0.120       # Canada
                 }):
        self.measurement_interval = measurement_interval
        self.lock = Lock()
        self.setup_logging()
        
        # Energy calculation parameters
        self.pue = pue
        self.cpu_idle_power = cpu_idle_power
        self.cpu_max_power = cpu_max_power
        self.ram_idle_power_per_gb = ram_idle_power_per_gb
        self.ram_active_power_per_gb = ram_active_power_per_gb
        self.disk_power = disk_power
        self.k_read = k_read
        self.k_write = k_write
        self.co2_factors = co2_factors

    def calculate_co2_emissions(self, power_metrics, region='global', duration_hours=1):
        """
        Calculate CO2 emissions based on power consumption and region.
        
        Parameters:
        power_metrics (dict): Dictionary containing power consumption metrics
        region (str): Region code for CO2 emission factor
        duration_hours (float): Duration in hours for the calculation
        
        Returns:
        dict: CO2 emissions metrics in kg CO2e
        """
        try:
            # Get region-specific CO2 factor (kg CO2 per kWh)
            co2_factor = self.co2_factors.get(region.lower(), self.co2_factors['global'])
            
            # Convert power (Watts) to energy (kWh) for the specified duration
            kwh_factor = duration_hours / 1000  # Convert W to kW and multiply by hours
            
            # Calculate CO2 emissions for each component
            cpu_emissions = power_metrics['cpu_power'] * kwh_factor * co2_factor
            ram_emissions = power_metrics['ram_power'] * kwh_factor * co2_factor
            disk_base_emissions = power_metrics['disk_base_power'] * kwh_factor * co2_factor
            disk_io_emissions = power_metrics['disk_io_power'] * kwh_factor * co2_factor
            
            # Calculate total emissions
            total_emissions = (cpu_emissions + ram_emissions + 
                             disk_base_emissions + disk_io_emissions)
            
            return {
                'total_co2': total_emissions,
                'cpu_co2': cpu_emissions,
                'ram_co2': ram_emissions,
                'disk_base_co2': disk_base_emissions,
                'disk_io_co2': disk_io_emissions,
                'co2_factor': co2_factor,
                'region': region
            }
            
        except Exception as e:
            logging.error(f"Error calculating CO2 emissions: {e}")
            return None

    def calculate_energy_consumption(self, metrics):
        """
        Calculate energy consumption based on system metrics using the provided formula.
        Returns energy consumption in Watts.
        """
        try:
            # CPU Power
            cpu_power = (self.cpu_idle_power + 
                        (self.cpu_max_power - self.cpu_idle_power) * 
                        metrics['cpu_percent'] / 100)

            # RAM Power
            ram_power = metrics['ram_total'] * (
                self.ram_idle_power_per_gb + 
                (self.ram_active_power_per_gb - self.ram_idle_power_per_gb) * 
                metrics['ram_percent'] / 100
            )            

            # Disk Base Power
            disk_base_power = metrics['storage_device_count'] * self.disk_power            

            # Disk I/O Power
            disk_io_power = (
                self.k_read * metrics['disk_read_bytes'] +
                self.k_write * metrics['disk_write_bytes']
            )            

            # Total Power
            total_power = self.pue * (
                cpu_power + ram_power + disk_base_power + disk_io_power
            )

            return {
                'total_power': total_power,
                'cpu_power': cpu_power,
                'ram_power': ram_power,
                'disk_base_power': disk_base_power,
                'disk_io_power': disk_io_power
            }

        except Exception as e:
            logging.error(f"Error calculating energy consumption: {e}")
            return None

    def setup_logging(self):
        """Set up logging for the program."""
        logging.basicConfig(
            filename='remote_monitor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def execute_ssh_command(self, ssh, command):
        """Execute command over SSH."""
        stdin, stdout, stderr = ssh.exec_command(command)
        return stdout.read().decode('utf-8').strip()

    def detect_os_type(self, ssh):
        """Detect if the remote server is Windows or Linux using SSH."""
        try:
            # Try Linux command
            result = self.execute_ssh_command(ssh, "uname -s")
            if result.strip():
                return 'Linux'
        except:
            pass

        try:
            # Try Windows PowerShell command
            result = self.execute_ssh_command(ssh, "powershell -Command $PSVersionTable")
            if "PSVersion" in result:
                return 'Windows'
        except:
            pass

        raise Exception("Unable to detect OS type. Ensure SSH is configured correctly.")

    def get_linux_metrics(self, ssh):
        """Collect metrics from Linux server."""
        metrics = {}

        # Get OS Version
        try:
            cmd = "cat /etc/os-release | grep -E '^VERSION=|^NAME=' | awk -F'\"' '{print $2}' | paste -sd ' '"
            os_info = self.execute_ssh_command(ssh, cmd)
            metrics['os_version'] = os_info
        except:
            metrics['os_version'] = "Unknown"

        # Get Storage Devices
        try:
            cmd = "lsblk -d -o NAME,TYPE,SIZE | grep -E 'disk' | wc -l"
            disk_count = int(self.execute_ssh_command(ssh, cmd))
            
            cmd = "lsblk -d -o NAME,TYPE,SIZE | grep -E 'disk' | awk '{print $1, $3}'"
            disk_details = self.execute_ssh_command(ssh, cmd)
            
            metrics['storage_device_count'] = disk_count
            metrics['storage_devices'] = disk_details
        except:
            metrics['storage_device_count'] = 0
            metrics['storage_devices'] = "Unknown"

        # CPU Usage
        try:
            cmd = "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'"
            metrics['cpu_percent'] = float(self.execute_ssh_command(ssh, cmd))
        except:
            metrics['cpu_percent'] = 0

        # RAM Metrics
        try:
            cmd = "free -b | grep Mem | awk '{print $2,$3,$3/$2 * 100}'"
            output = self.execute_ssh_command(ssh, cmd).split()
            metrics['ram_total'] = float(output[0]) / (1024 ** 3)
            metrics['ram_used'] = float(output[1]) / (1024 ** 3)
            metrics['ram_percent'] = float(output[2])
        except:
            metrics['ram_total'] = metrics['ram_used'] = metrics['ram_percent'] = 0

        # Disk I/O
        try:
            cmd = "cat /proc/diskstats | awk '{print $3}' | grep -E '^sd' | head -1"
            disk_name = self.execute_ssh_command(ssh, cmd)
            cmd = f"cat /proc/diskstats | grep -w '{disk_name}' | awk '{{print $6,$10}}'"
            output = self.execute_ssh_command(ssh, cmd).split()
            metrics['disk_read_bytes'] = float(output[0]) * 512
            metrics['disk_write_bytes'] = float(output[1]) * 512
        except:
            metrics['disk_read_bytes'] = metrics['disk_write_bytes'] = 0

        return metrics

    def get_windows_metrics(self, ssh):
        """Collect metrics from Windows server using PowerShell over SSH."""
        metrics = {}

        # Get OS Version
        try:
            os_cmd = 'powershell -Command "(Get-WmiObject -Class Win32_OperatingSystem).Caption"'
            metrics['os_version'] = self.execute_ssh_command(ssh, os_cmd).strip()
        except:
            metrics['os_version'] = "Unknown"

        # Get Storage Devices
        try:
            disk_cmd = 'powershell -Command "& { $disks = Get-WmiObject -Class Win32_DiskDrive; Write-Output $disks.Count; $disks | ForEach-Object { $_.DeviceID + \' \' + [math]::Round($_.Size/1GB, 2) + \'GB\' } }"'
            raw_output = self.execute_ssh_command(ssh, disk_cmd)
            disk_lines = raw_output.strip().split('\n')
            
            metrics['storage_device_count'] = int(disk_lines[0])
            metrics['storage_devices'] = '; '.join(disk_lines[1:])
        except Exception as e:
            logging.error(f"Error getting Windows storage devices: {e}")
            metrics['storage_device_count'] = 0
            metrics['storage_devices'] = "Unknown"

        # CPU Usage
        try:
            cpu_cmd = 'powershell -Command "(Get-WmiObject -Class win32_processor | Measure-Object -Property LoadPercentage -Average).Average"'
            metrics['cpu_percent'] = float(self.execute_ssh_command(ssh, cpu_cmd))
        except:
            metrics['cpu_percent'] = 0

        # RAM Metrics
        try:
            ram_cmd = 'powershell -Command "& {$os = Get-WmiObject -Class win32_operatingsystem; $total = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2); $free = [math]::Round($os.FreePhysicalMemory / 1MB, 2); $used = $total - $free; $percent = [math]::Round(($used / $total) * 100, 2); Write-Output \"$total,$used,$percent\"}" '
            raw_output = self.execute_ssh_command(ssh, ram_cmd)
            ram_output = raw_output.replace('\r\n', ',').split(',')
            ram_output = [float(x.strip()) for x in ram_output]
            metrics['ram_total'] = float(ram_output[0])
            metrics['ram_used'] = float(ram_output[1])
            metrics['ram_percent'] = float(ram_output[2])
        except Exception as e:
            logging.error(f"Error parsing Windows RAM metrics: {e}")
            metrics['ram_total'] = metrics['ram_used'] = metrics['ram_percent'] = 0

        # Disk I/O
        try:
            disk_cmd = 'powershell -Command "& { (Get-WmiObject -Class Win32_PerfFormattedData_PerfDisk_LogicalDisk | Where-Object { $_.Name -eq \'C:\' }) | Select-Object -ExpandProperty DiskReadBytesPersec; (Get-WmiObject -Class Win32_PerfFormattedData_PerfDisk_LogicalDisk | Where-Object { $_.Name -eq \'C:\' }) | Select-Object -ExpandProperty DiskWriteBytesPersec }"'
            raw_output = self.execute_ssh_command(ssh, disk_cmd)
            disk_output = raw_output.replace('\r\n', ',').split(',')
            disk_output = [float(x.strip()) for x in disk_output]
            metrics['disk_read_bytes'] = float(disk_output[0])
            metrics['disk_write_bytes'] = float(disk_output[1])
        except Exception as e:
            logging.error(f"Error parsing Windows Disk metrics: {e}")
            metrics['disk_read_bytes'] = metrics['disk_write_bytes'] = 0

        return metrics

    def get_remote_metrics(self, hostname, username, password):
        """Collect metrics from a remote server using SSH."""
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname, username=username, password=password, timeout=10)

            os_type = self.detect_os_type(ssh)
            metrics = self.get_linux_metrics(ssh) if os_type == 'Linux' else self.get_windows_metrics(ssh)
            ssh.close()

            # Add common metrics with a formatted timestamp
            metrics.update({
                'timestamp': datetime.now(),
                'hostname': hostname,
                'os_type': os_type
            })

            # Calculate and add energy metrics
            energy_metrics = self.calculate_energy_consumption(metrics)
            if energy_metrics:
                metrics.update(energy_metrics)
                
                # Calculate and add CO2 emissions
                # Convert measurement interval to hours
                duration_hours = self.measurement_interval / 3600
                co2_metrics = self.calculate_co2_emissions(
                    energy_metrics, 
                    region='global',
                    duration_hours=duration_hours
                )
                if co2_metrics:
                    metrics.update(co2_metrics)

            return metrics

        except Exception as e:
            logging.error(f"Error collecting metrics from {hostname}: {e}")
            return None

    def monitor_servers(self, server_list, duration_seconds=60):
        """Monitor multiple servers concurrently."""
        all_measurements = []
        start_time = time.time()

        def monitor_single_server(server_info):
            server_measurements = []
            while time.time() - start_time < duration_seconds:
                metrics = self.get_remote_metrics(
                    server_info['ip'],
                    server_info['username'],
                    server_info['password']
                )
                if metrics:
                    server_measurements.append(metrics)
                time.sleep(self.measurement_interval)
            return server_measurements

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(server_list)) as executor:
            future_to_server = {
                executor.submit(monitor_single_server, server): server
                for server in server_list
            }

            for future in concurrent.futures.as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    measurements = future.result()
                    all_measurements.extend(measurements)
                except Exception as e:
                    logging.error(f"Error monitoring server {server['ip']}: {e}")

        return pd.DataFrame(all_measurements)


def read_server_credentials(excel_file):
    """Read server credentials from Excel file."""
    try:
        df = pd.read_excel(excel_file)
        required_columns = ['ip', 'username', 'password']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Excel file must contain columns: ip, username, password")

        return df.to_dict('records')
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        return None


def main():
    # Excel file path
    excel_file = "server_credentials.xlsx"

    if not os.path.exists(excel_file):
        logging.error(f"Error: {excel_file} not found!")
        return

    server_list = read_server_credentials(excel_file)
    if not server_list:
        return

    monitor = RemoteSystemMonitor(
        measurement_interval=1,
        pue=1.67,
        cpu_idle_power=10,
        cpu_max_power=100,
        ram_idle_power_per_gb=0.3,
        ram_active_power_per_gb=0.6,
        disk_power=8,
        k_read=8e-8,
        k_write=8e-8
    )

    print(f"Starting monitoring of {len(server_list)} servers for 60 seconds...")
    df = monitor.monitor_servers(server_list, duration_seconds=60)

    # Generate summary statistics per server
    print("\nSummary Statistics per Server:")
    for hostname in df['hostname'].unique():
        server_df = df[df['hostname'] == hostname]
        os_type = server_df['os_type'].iloc[0]
        os_version = server_df['os_version'].iloc[0]
        storage_count = server_df['storage_device_count'].iloc[0]
        storage_details = server_df['storage_devices'].iloc[0]
        region = server_df['region'].iloc[0]
        
        print(f"\nServer: {hostname}")
        print(f"OS Type: {os_type}")
        print(f"OS Version: {os_version}")
        print(f"Storage Devices Count: {storage_count}")
        print(f"Storage Devices: {storage_details}")
        print(f"Region: {region}")
        print("\nResource Usage:")
        print(f"Average CPU Usage: {server_df['cpu_percent'].mean():.2f}%")
        print(f"Average RAM Usage: {server_df['ram_percent'].mean():.2f}%")
        
        print("\nPower Consumption:")
        print(f"Average Total Power: {server_df['total_power'].mean():.2f} Watts")
        print(f"Average CPU Power: {server_df['cpu_power'].mean():.2f} Watts")
        print(f"Average RAM Power: {server_df['ram_power'].mean():.2f} Watts")
        print(f"Average Disk Power: {(server_df['disk_base_power'] + server_df['disk_io_power']).mean():.2f} Watts")
        
        print("\nCO2 Emissions:")
        print(f"CO2 Emission Factor: {server_df['co2_factor'].iloc[0]:.3f} kg CO2/kWh")
        print(f"Total CO2 Emissions: {server_df['total_co2'].sum():.6f} kg CO2e")
        print(f"CPU CO2 Emissions: {server_df['cpu_co2'].sum():.6f} kg CO2e")
        print(f"RAM CO2 Emissions: {server_df['ram_co2'].sum():.6f} kg CO2e")
        print(f"Disk CO2 Emissions: {(server_df['disk_base_co2'] + server_df['disk_io_co2']).sum():.6f} kg CO2e")

    # Save results to CSV with append mode
    filename = os.path.join(RESULT_DIR, "multiple_server_data.csv")
    
    # Write header only if file doesn't exist
    df.to_csv(filename, 
              mode='a', 
              header=not os.path.exists(filename), 
              index=False)
    
    print(f"\nDetailed metrics appended to {filename}")

if __name__ == "__main__":
    main()
