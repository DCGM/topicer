import subprocess

# Spustí příkaz 'ls -l' (na Linuxu/macOS) nebo 'dir' (na Windows)
# V tomto příkladu použijeme obecný příkaz
try:
    result = subprocess.run(['echo', 'Ahoj ze shellu!'], 
                            capture_output=True, 
                            text=True, 
                            check=True) # check=True vyvolá chybu, pokud je návratový kód != 0
    
    print(f"Návratový kód: {result.returncode}")
    print("--- Standardní výstup (stdout) ---")
    print(result.stdout)
    
except subprocess.CalledProcessError as e:
    print(f"Chyba při spouštění příkazu: {e}")
    print(f"Standardní chybový výstup (stderr): {e.stderr}")
    
except FileNotFoundError:
    print("Příkaz nebyl nalezen (např. 'echo' neexistuje v PATH).")