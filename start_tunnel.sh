#!/bin/bash

# Zkontroluje, zda existuje konfigurační soubor a načte ho
CONFIG_FILE="config.sh"
if [ -f "$CONFIG_FILE" ]; then
    echo "Načítám konfiguraci z $CONFIG_FILE..."
    source "$CONFIG_FILE"
else
    echo "Chyba: Konfigurační soubor $CONFIG_FILE nebyl nalezen."
    exit 1
fi

# Sestavení prvního tunelu (-L L_PORT:T_HOST:T_PORT)
TUNNEL_1="-L ${TUNNEL_1_LOCAL}:${TUNNEL_1_TARGET_HOST}:${TUNNEL_1_TARGET_PORT}"

# Sestavení druhého tunelu
TUNNEL_2="-L ${TUNNEL_2_LOCAL}:${TUNNEL_2_TARGET_HOST}:${TUNNEL_2_TARGET_PORT}"

# Sestavení celého příkazu ssh
SSH_COMMAND="ssh ${TUNNEL_1} ${TUNNEL_2} ${SSH_USER}@${SSH_SERVER}"

echo "--- Spouštím SSH tunel ---"
echo "Příkaz: ${SSH_COMMAND}"

# Spuštění SSH příkazu
# '-N' zabrání spuštění vzdáleného příkazu (jen port forwarding)
# '-C' pro kompresi (volitelné, ale doporučené)
# '-v' pro verbose výstup (můžete smazat, ale pomůže při ladění)

# Pokud chcete, aby skript běžel v popředí a čekal:
# $SSH_COMMAND -N -C

# Pokud chcete, aby se SSH tunel spustil na pozadí a vy se vrátili k příkazové řádce:
echo "Spouštím na pozadí..."
$SSH_COMMAND -N -C -f

# Kontrola, zda se SSH proces spustil
if [ $? -eq 0 ]; then
    echo "✅ SSH tunely by měly běžet na pozadí."
    echo "Tunely: $TUNNEL_1 a $TUNNEL_2"
else
    echo "❌ Chyba při spouštění SSH."
fi