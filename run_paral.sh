#!/bin/bash

# Script to run multiple Python commands in parallel (up to max_parallel),
# and keep THE SYSTEM CPU busy if overall usage drops too low.

# --- Configuration ---
commands=(
    "python Main_Association_Rule.py --file_type MiraiBotnet --association RARM" # Example 1
    "python Main_Association_Rule.py --file_type MiraiBotnet --association Eclat" # Example 2
    "python Main_Association_Rule.py --file_type MiraiBotnet --association SaM"   # Example 3
    # Add your actual commands here
)

NUM_CORES=$(nproc)
max_parallel=$NUM_CORES 
# max_parallel=2 # Or set a fixed number

SYSTEM_CPU_USAGE_THRESHOLD=10 

IDLE_CPU_COMMAND="yes > /dev/null"
CHECK_INTERVAL=5
# --- End Configuration ---

system_idle_pid=""

start_system_idle_spin() {
    if [ -z "$system_idle_pid" ] || ! ps -p "$system_idle_pid" > /dev/null; then
        echo "[SysKeeper] System CPU usage low. Starting system-wide idle spin..."
        eval "$IDLE_CPU_COMMAND" &
        system_idle_pid=$!
        echo "[SysKeeper] System idle spin process started with PID: $system_idle_pid"
    fi
}

stop_system_idle_spin() {
    if [ -n "$system_idle_pid" ] && ps -p "$system_idle_pid" > /dev/null; then
        echo "[SysKeeper] System CPU usage recovered or all tasks ended. Stopping system idle spin (PID: $system_idle_pid)..."
        pkill -P "$system_idle_pid" > /dev/null 2>&1
        kill "$system_idle_pid" > /dev/null 2>&1
        sleep 0.1
        if ps -p "$system_idle_pid" > /dev/null; then
            kill -9 "$system_idle_pid" > /dev/null 2>&1
        fi
        wait "$system_idle_pid" 2>/dev/null
        system_idle_pid=""
        echo "[SysKeeper] System idle spin process stopped."
    fi
}

monitor_system_cpu_and_manage_load() {
    echo "[SysKeeper] Starting system-wide CPU usage monitoring (will stop when main script tasks end)."
    
    while [ "$main_tasks_running" = true ]; do
        # Try mpstat first, if fails or not available, try top
        current_system_cpu_usage=$(mpstat 1 1 2>/dev/null | awk 'END{print 100 - $NF}')
        current_system_cpu_usage=${current_system_cpu_usage%.*} # Get integer part

        if ! [[ "$current_system_cpu_usage" =~ ^[0-9]+$ ]]; then # Check if not a valid number (e.g., mpstat failed)
            # Fallback to top (more portable but can be slower/heavier)
            # This gets the CPU idle percentage and subtracts from 100.
            idle_cpu=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/")
            if [[ "$idle_cpu" =~ ^[0-9.]+$ ]]; then
                current_system_cpu_usage=$(echo "100 - $idle_cpu" | bc)
                current_system_cpu_usage=${current_system_cpu_usage%.*} # Integer part
            else
                echo "[SysKeeper] Could not retrieve system CPU usage from mpstat or top. Assuming high usage."
                current_system_cpu_usage=100 # Default to high usage if unknown
            fi
        fi

        echo "[SysKeeper] Current system-wide CPU usage: $current_system_cpu_usage%"

        if (( current_system_cpu_usage < SYSTEM_CPU_USAGE_THRESHOLD )); then
            start_system_idle_spin
        else
            stop_system_idle_spin
        fi
        
        # Check if any of the main python scripts are still running under this script's process tree
        # If not, and main_tasks_running is still true, it means all tasks finished before the main loop knew.
        if ! pgrep -P $$ python > /dev/null && [ "$main_tasks_running" = true ]; then
            echo "[SysKeeper] Detected no running python child processes. Signaling monitor to stop."
            main_tasks_running=false # Signal to exit the while loop
        fi
        sleep "$CHECK_INTERVAL"
    done

    echo "[SysKeeper] System-wide CPU usage monitoring ended."
    stop_system_idle_spin 
}

# --- Main Execution ---
running_main_procs=0
main_pids=() 

main_tasks_running=true 
monitor_system_cpu_and_manage_load &
SYSTEM_MONITOR_PID=$!
echo "System CPU Keeper process started with PID: $SYSTEM_MONITOR_PID"
sleep 1 

job_pids=() # Array to store PIDs of the launched main commands

cleanup() {
    echo "[MainScript] Cleaning up..."
    main_tasks_running=false # Signal monitor to stop

    if ps -p "$SYSTEM_MONITOR_PID" > /dev/null; then
        echo "[MainScript] Stopping system monitor (PID: $SYSTEM_MONITOR_PID)..."
        kill "$SYSTEM_MONITOR_PID" > /dev/null 2>&1 &
    fi
    stop_system_idle_spin 

    echo "[MainScript] Terminating any remaining main tasks..."
    for pid_to_kill in "${job_pids[@]}"; do
        if ps -p "$pid_to_kill" > /dev/null; then
            echo "[MainScript] Terminating task PID: $pid_to_kill"
            # Send SIGTERM first, then SIGKILL if necessary, and to process group
            kill -- -"$(ps -o pgid= -p "$pid_to_kill" | grep -o '[0-9]*')" > /dev/null 2>&1
            sleep 0.5
            if ps -p "$pid_to_kill" > /dev/null; then
                 kill -9 -- -"$(ps -o pgid= -p "$pid_to_kill" | grep -o '[0-9]*')" > /dev/null 2>&1
            fi
        fi
    done
    wait
    echo "[MainScript] Cleanup finished."
    # exit # Exit trap should not call exit itself usually
}
trap cleanup EXIT SIGINT SIGTERM

for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    
    while (( running_main_procs >= max_parallel )); do
        # Wait for any of the running jobs to finish
        echo "[MainScript] Max parallel processes ($max_parallel) reached. Waiting for a slot..."
        
        # Check for finished PIDs more reliably
        finished_pid=0
        while true; do
            for ((j=0; j<${#job_pids[@]}; j++)); do
                pid_to_check="${job_pids[$j]}"
                if ! ps -p "$pid_to_check" > /dev/null; then # Process no longer exists
                    finished_pid=$pid_to_check
                    # Remove from job_pids array
                    job_pids=("${job_pids[@]:0:$j}" "${job_pids[@]:$(($j + 1))}")
                    ((running_main_procs--))
                    echo "[MainScript] Job with PID $finished_pid finished. Slots available: $((max_parallel - running_main_procs))"
                    break 2 # Break both loops
                fi
            done
            if (( running_main_procs < max_parallel )); then # Slot became free without explicit finish detection (e.g. external kill)
                break # Break inner while loop, outer will re-evaluate
            fi
            sleep 1 # Check every second
        done
        if (( finished_pid == 0 )) && (( running_main_procs >= max_parallel )); then
             # This case should ideally not be hit if the loop above works, but as a fallback:
             echo "[MainScript] Fallback: Waiting for any process to finish using wait -n..."
             wait -n
             # Re-evaluate running_main_procs (less precise who finished)
             temp_running_count=0
             temp_new_job_pids=()
             for pid_check_in_fallback in "${job_pids[@]}"; do 
                 if ps -p "$pid_check_in_fallback" > /dev/null; then 
                     ((temp_running_count++)); temp_new_job_pids+=("$pid_check_in_fallback"); 
                 fi; 
             done
             if (( temp_running_count < running_main_procs )); then
                echo "[MainScript] A job finished (detected by wait -n). Running: $temp_running_count"
             fi
             running_main_procs=$temp_running_count
             job_pids=("${temp_new_job_pids[@]}")
        fi
    done
    
    echo "----------------------------------------------------------------------"
    echo "[MainScript] Starting job $((i + 1))/${#commands[@]}: $cmd"
    # Execute in a subshell to get a new PGID for easier cleanup
    (trap 'echo "Subshell for PID $$ killed"; exit' SIGTERM SIGINT; eval "$cmd") &
    CURRENT_MAIN_PID=$!
    job_pids+=("$CURRENT_MAIN_PID") 
    ((running_main_procs++))
    echo "[MainScript] Job started with PID: $CURRENT_MAIN_PID. Running processes: $running_main_procs"
done

echo "[MainScript] All commands have been launched. Waiting for remaining $running_main_procs jobs to complete..."

while (( running_main_procs > 0 )); do
    finished_pid_final=0
    # Similar PID check loop as above
    for ((k=0; k<${#job_pids[@]}; k++)); do
        pid_to_check_final="${job_pids[$k]}"
        if ! ps -p "$pid_to_check_final" > /dev/null; then
            finished_pid_final=$pid_to_check_final
            job_pids=("${job_pids[@]:0:$k}" "${job_pids[@]:$(($k + 1))}")
            ((running_main_procs--))
            echo "[MainScript] Final wait: Job with PID $finished_pid_final finished. Remaining: $running_main_procs"
            break # Found a finished one, re-iterate outer while
        fi
    done
    if (( finished_pid_final == 0 )) && (( running_main_procs > 0 )); then
        # If no PID was found finished by ps check, but some should be running, wait a bit.
        sleep 1 
    elif (( running_main_procs == 0 )); then
        break # All done
    fi
    # If finished_pid_final was > 0, the outer while loop will re-evaluate running_main_procs
    # if it was 0 and procs > 0, it means all are still running, so sleep and re-check ps
done

echo "[MainScript] All main Python scripts completed." 
main_tasks_running=false # Signal system monitor to stop

# Wait for the system monitor to stop gracefully
echo "[MainScript] Waiting for system monitor (PID: $SYSTEM_MONITOR_PID) to stop..."
if ps -p "$SYSTEM_MONITOR_PID" > /dev/null; then
    wait "$SYSTEM_MONITOR_PID" 2>/dev/null
fi

cleanup # Call cleanup one last time to be sure

echo "[MainScript] All tasks finished." 
exit 0
