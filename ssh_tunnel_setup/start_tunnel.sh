#!/bin/bash

# Checks if the configuration file exists and sources it
CONFIG_FILE="config.sh"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE..."
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file $CONFIG_FILE not found."
    exit 1
fi

# Construction of the first tunnel
TUNNEL_1="-L ${TUNNEL_1_LOCAL}:${TUNNEL_1_TARGET_HOST}:${TUNNEL_1_TARGET_PORT}"

# Construction of the second tunnel
TUNNEL_2="-L ${TUNNEL_2_LOCAL}:${TUNNEL_2_TARGET_HOST}:${TUNNEL_2_TARGET_PORT}"

# Construction of the third tunnel
TUNNEL_3="-L ${TUNNEL_3_LOCAL}:${TUNNEL_3_TARGET_HOST}:${TUNNEL_3_TARGET_PORT}"

# Construction of the entire ssh command
SSH_COMMAND="ssh ${TUNNEL_1} ${TUNNEL_2} ${TUNNEL_3} ${SSH_USER}@${SSH_SERVER}"

echo "--- Starting SSH tunnels ---"
echo "Command: ${SSH_COMMAND}"

# SSH command execution
# '-N' prevents remote command execution (only port forwarding)
# '-C' enables compression (optional but recommended)
# '-v' for verbose output (can be removed but helps with debugging)

# If you want the script to run in the foreground and wait:
# $SSH_COMMAND -N -C

# If you want the SSH tunnel to run in the background and return to the command line:
echo "Running in the background..."
$SSH_COMMAND -N -C -f # -v

# Check if the SSH process started successfully
if [ $? -eq 0 ]; then
    echo "✅ SSH tunnels should be running in the background."
    echo "Tunnels: $TUNNEL_1, $TUNNEL_2, and $TUNNEL_3"
else
    echo "❌ Error while starting SSH."
fi