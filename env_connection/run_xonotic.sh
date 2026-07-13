#!/bin/bash
# Placeholder script to launch Xonotic and start the RL interface
echo "Launching Xonotic..."
# In a real setup, you'd call the Xonotic binary with appropriate flags
# e.g., ./xonotic-linux64 -game xonotic +set net_port 27960 &
# Then start the Python bridge
python3 $(dirname "$0")/../event_recording/record_events.py &
