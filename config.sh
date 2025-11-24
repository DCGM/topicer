# --- KONFIGURACE SSH TUNELU ---

# Uživatelské jméno na vzdáleném serveru
SSH_USER="xjuric31"

# Adresa vzdáleného serveru
SSH_SERVER="semant.cz"

# První tunel: LOKÁLNÍ_PORT:CÍLOVÁ_HOST:CÍLOVÝ_PORT
TUNNEL_1_LOCAL="9000"
TUNNEL_1_TARGET_HOST="localhost"
TUNNEL_1_TARGET_PORT="8080"

# Druhý tunel: LOKÁLNÍ_PORT:CÍLOVÁ_HOST:CÍLOVÝ_PORT
TUNNEL_2_LOCAL="50055"
TUNNEL_2_TARGET_HOST="localhost"
TUNNEL_2_TARGET_PORT="50051"