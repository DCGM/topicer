import subprocess
import configparser

# 1. Načtení konfigurace (např. ze souboru INI)
config = configparser.ConfigParser()
config.read('config.ini')

user = config['SSH']['user']
server = config['SSH']['server']
tunnel_1 = config['TUNNEL_1']
tunnel_2 = config['TUNNEL_2']

# 2. Sestavení příkazu
# -L L_PORT:T_HOST:T_PORT
tunnel_1_arg = f"-L {tunnel_1['local_port']}:{tunnel_1['target_host']}:{tunnel_1['target_port']}"
tunnel_2_arg = f"-L {tunnel_2['local_port']}:{tunnel_2['target_host']}:{tunnel_2['target_port']}"

ssh_command = [
    'ssh',
    '-N',
    '-C',
    '-f',
    tunnel_1_arg,
    tunnel_2_arg,
    f"{user}@{server}"
]

# 3. Spuštění příkazu
try:
    print(f"Spouštím: {' '.join(ssh_command)}")
    # Spustí SSH příkaz. check=True zajistí chybu při neúspěchu.
    subprocess.run(ssh_command, check=True)
    print("✅ SSH tunely spuštěny na pozadí.")
except subprocess.CalledProcessError as e:
    print(f"❌ Chyba při spouštění SSH: {e}")