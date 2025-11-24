import subprocess
import configparser

# 1. Load configuration (e.g., from an INI file)
config = configparser.ConfigParser()
config.read('config.ini')

user = config['SSH']['user']
server = config['SSH']['server']
tunnel_1 = config['TUNNEL_1']
tunnel_2 = config['TUNNEL_2']


# 2. Building the command
# -L L_PORT:T_HOST:T_PORT
tunnel_1_arg = f"-L {tunnel_1['local_port']}:{tunnel_1['target_host']}:{tunnel_1['target_port']}"
tunnel_2_arg = f"-L {tunnel_2['local_port']}:{tunnel_2['target_host']}:{tunnel_2['target_port']}"

ssh_command = [
    'ssh',
    '-N',
    '-C',
    '-f',
    '-v',
    tunnel_1_arg,
    tunnel_2_arg,
    f"{user}@{server}"
]

# 3. Running the command
try:
    print(f"Running: {' '.join(ssh_command)}")
    # Runs the SSH command. check=True ensures an error is raised on failure.
    subprocess.run(ssh_command, check=True)
    print("✅ SSH tunnels should be running in the background.")
except subprocess.CalledProcessError as e:
    print(f"❌ Error while starting SSH: {e}")