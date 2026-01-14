#!/bin/bash

# Script to run multiple Python commands sequentially, and for each command,
# keep the CPU busy if its usage drops to prevent session termination.

# --- Configuration ---
# Define your list of commands to be executed sequentially.
# Each command will be monitored individually.
commands=(
    "python Main_Association_Rule.py --file_type MiraiBotnet --association RARM" # Example 1: Replace with your actual command
    "python Main_Association_Rule.py --file_type MiraiBotnet --association Eclat" # Example 2: Replace with your actual command
    # "python Wildfire_spread_graph.py --data_number 1" # Add more commands as needed
    # "python Wildfire_spread_graph.py --data_number 2"
)

# CPU usage threshold (integer percentage). If Python script's CPU usage is below this, start idle spin.
CPU_USAGE_THRESHOLD=10

# Command to generate CPU load for idle spin. 'yes > /dev/null' uses one core at 100%.
IDLE_CPU_COMMAND="yes > /dev/null"

# How often to check CPU usage (in seconds).
CHECK_INTERVAL=5
# --- End Configuration ---

idle_pid="" # PID of the idle spin process

# Function to start the idle spin process
start_idle_spin() {
    if [ -z "$idle_pid" ] || ! ps -p "$idle_pid" > /dev/null; then
        echo "[Keeper] Main script CPU usage low. Starting idle spin..."
        eval "$IDLE_CPU_COMMAND" &
        idle_pid=$!
        echo "[Keeper] Idle spin process started with PID: $idle_pid"
    # else
        # echo "[Keeper] Idle spin already running (PID: $idle_pid)."
    fi
}

# Function to stop the idle spin process
stop_idle_spin() {
    if [ -n "$idle_pid" ] && ps -p "$idle_pid" > /dev/null; then
        echo "[Keeper] Main script CPU usage recovered or script ended. Stopping idle spin (PID: $idle_pid)..."
        pkill -P "$idle_pid" 
        kill "$idle_pid" > /dev/null 2>&1
        sleep 0.1 
        if ps -p "$idle_pid" > /dev/null; then 
            kill -9 "$idle_pid" > /dev/null 2>&1 
        fi
        wait "$idle_pid" 2>/dev/null 
        idle_pid=""
        echo "[Keeper] Idle spin process stopped."
    fi
}

# Function to monitor the main script's CPU usage and manage idle spin
monitor_and_manage_cpu_load() {
    local main_script_pid=$1
    local command_being_monitored="$2"
    local last_cpu_above_threshold=true # Flag to track previous state, true means CPU was high or initial state
    # local first_check=true # Alternative for logging initial state

    echo "[Keeper] Monitoring CPU usage for: $command_being_monitored (PID: $main_script_pid)"

    while ps -p "$main_script_pid" > /dev/null; do
        current_cpu_usage=$(ps -p "$main_script_pid" -o %cpu --no-headers | awk '{print int($1)}')
        if [ -z "$current_cpu_usage" ]; then # PID disappeared or couldn't read CPU usage
            # Log this event as it's important
            echo "[Keeper] Could not retrieve CPU usage for PID $main_script_pid ($command_being_monitored) or PID disappeared."
            # Attempt to stop idle spin if it was running
            if [ -n "$idle_pid" ] && ps -p "$idle_pid" > /dev/null; then
                 stop_idle_spin # stop_idle_spin will print its logs
            fi
            break # Exit monitoring loop
        fi

        # Periodic CPU usage logs can be uncommented if needed (for debugging)
        # echo "[Keeper] Debug: Script '$command_being_monitored' (PID: $main_script_pid) current CPU: $current_cpu_usage% (Threshold: $CPU_USAGE_THRESHOLD, Idle PID: $idle_pid)"

        if (( current_cpu_usage < CPU_USAGE_THRESHOLD )); then
            # CPU is low. Start spin only if it was previously high or if idle_pid is not running.
            if [ "$last_cpu_above_threshold" = true ] || ! ( [ -n "$idle_pid" ] && ps -p "$idle_pid" > /dev/null ); then
                start_idle_spin # This function now logs "Starting idle spin..."
            fi
            last_cpu_above_threshold=false
        else
            # CPU is high (or recovered). Stop spin only if it was previously low and idle_pid is running.
            if [ "$last_cpu_above_threshold" = false ] && ( [ -n "$idle_pid" ] && ps -p "$idle_pid" > /dev/null ); then
                stop_idle_spin # This function now logs "Stopping idle spin..."
            fi
            last_cpu_above_threshold=true
        fi
        sleep "$CHECK_INTERVAL"
    done

    echo "[Keeper] Script '$command_being_monitored' (PID: $main_script_pid) appears to have finished."
    # Ensure idle spin is stopped finally, will log if it was running
    if [ -n "$idle_pid" ] && ps -p "$idle_pid" > /dev/null; then
        stop_idle_spin 
    fi
    echo "[Keeper] CPU monitoring for PID $main_script_pid ($command_being_monitored) ended."
}

# --- Main Execution ---
overall_exit_code=0

for cmd in "${commands[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Starting Python script: $cmd"
    # Execute the Python command in the background
    eval "$cmd" &
    MAIN_SCRIPT_PID=$!
    echo "Python script started with PID: $MAIN_SCRIPT_PID"

    # Start the CPU monitor and load manager in the background
    monitor_and_manage_cpu_load "$MAIN_SCRIPT_PID" "$cmd" &
    MONITOR_PID=$!
    echo "CPU keeper process for '$cmd' started with PID: $MONITOR_PID"

    # Wait for the current Python script to complete
    echo "Waiting for Python script (PID: $MAIN_SCRIPT_PID) '$cmd' to complete..."
    wait "$MAIN_SCRIPT_PID"
    SCRIPT_EXIT_CODE=$?

    echo "Python script '$cmd' (PID: $MAIN_SCRIPT_PID) completed with exit code: $SCRIPT_EXIT_CODE."
    if (( SCRIPT_EXIT_CODE != 0 )); then
        overall_exit_code=$SCRIPT_EXIT_CODE
        echo "Warning: Command '$cmd' exited with error code $SCRIPT_EXIT_CODE."
    fi

    echo "Ensuring CPU keeper for '$cmd' (PID: $MONITOR_PID) is stopped..."
    if ps -p "$MONITOR_PID" > /dev/null; then
        kill "$MONITOR_PID" > /dev/null 2>&1 &
    fi
    # Call stop_idle_spin directly here to ensure the idle_pid associated with THIS cmd is stopped
    # The one inside monitor_and_manage_cpu_load might be for the next cmd if timing is tricky
    stop_idle_spin 
    echo "----------------------------------------------------------------------"
    echo ""

done

echo "All commands in the list have been processed."
echo "Overall script finished. Last non-zero exit code (if any): $overall_exit_code"
exit "$overall_exit_code"
