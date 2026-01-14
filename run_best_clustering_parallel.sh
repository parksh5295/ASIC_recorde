#!/bin/bash

# Script to run multiple clustering algorithm selection scripts in parallel

# --- Configuration ---
commands=(
    "python best_clustering_selector_parallel.py --file_type DARPA98 --file_number 1"
    "python best_clustering_selector_parallel.py --file_type CICIoT2023 --file_number 1"
    "python best_clustering_selector_parallel.py --file_type Kitsune --file_number 1"
    #"python best_clustering_selector_parallel.py --file_type IoTID20 --file_number 1"
    #"python best_clustering_selector_parallel.py --file_type netML --file_number 1"
    "python best_clustering_selector_parallel.py --file_type MiraiBotnet --file_number 1"
    "python best_clustering_selector_parallel.py --file_type CICIDS2017 --file_number 1"
    "python best_clustering_selector_parallel.py --file_type NSL-KDD --file_number 1"
    "python best_clustering_selector_parallel.py --file_type CICModbus23 --file_number 1"
)

# Check system resources
NUM_CORES=$(nproc)
max_parallel=$NUM_CORES  # Use all cores
# max_parallel=4  # or set a fixed number

SYSTEM_CPU_USAGE_THRESHOLD=10
IDLE_CPU_COMMAND="yes > /dev/null"
CHECK_INTERVAL=5

echo "=========================================="
echo "Best Clustering Algorithm Selector - Parallel"
echo "=========================================="
echo "System Resources:"
echo "  CPU Cores: $NUM_CORES"
echo "  Max Parallel Processes: $max_parallel"
echo "  Commands to Execute: ${#commands[@]}"
echo ""

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
    echo "[SysKeeper] Starting system-wide CPU usage monitoring..."
    
    while [ "$main_tasks_running" = true ]; do
        # Check CPU usage
        current_system_cpu_usage=$(mpstat 1 1 2>/dev/null | awk 'END{print 100 - $NF}')
        current_system_cpu_usage=${current_system_cpu_usage%.*}

        if ! [[ "$current_system_cpu_usage" =~ ^[0-9]+$ ]]; then
            idle_cpu=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/")
            if [[ "$idle_cpu" =~ ^[0-9.]+$ ]]; then
                current_system_cpu_usage=$(echo "100 - $idle_cpu" | bc)
                current_system_cpu_usage=${current_system_cpu_usage%.*}
            else
                echo "[SysKeeper] Could not retrieve system CPU usage. Assuming high usage."
                current_system_cpu_usage=100
            fi
        fi

        echo "[SysKeeper] Current system-wide CPU usage: $current_system_cpu_usage%"

        if (( current_system_cpu_usage < SYSTEM_CPU_USAGE_THRESHOLD )); then
            start_system_idle_spin
        else
            stop_system_idle_spin
        fi
        
        if ! pgrep -P $$ python > /dev/null && [ "$main_tasks_running" = true ]; then
            echo "[SysKeeper] Detected no running python child processes. Signaling monitor to stop."
            main_tasks_running=false
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
    main_tasks_running=false

    if ps -p "$SYSTEM_MONITOR_PID" > /dev/null; then
        echo "[MainScript] Stopping system monitor (PID: $SYSTEM_MONITOR_PID)..."
        kill "$SYSTEM_MONITOR_PID" > /dev/null 2>&1 &
    fi
    stop_system_idle_spin 

    echo "[MainScript] Terminating any remaining main tasks..."
    for pid_to_kill in "${job_pids[@]}"; do
        if ps -p "$pid_to_kill" > /dev/null; then
            echo "[MainScript] Terminating task PID: $pid_to_kill"
            kill -- -"$(ps -o pgid= -p "$pid_to_kill" | grep -o '[0-9]*')" > /dev/null 2>&1
            sleep 0.5
            if ps -p "$pid_to_kill" > /dev/null; then
                 kill -9 -- -"$(ps -o pgid= -p "$pid_to_kill" | grep -o '[0-9]*')" > /dev/null 2>&1
            fi
        fi
    done
    wait
    echo "[MainScript] Cleanup finished."
}
trap cleanup EXIT SIGINT SIGTERM

# Run each command in parallel
for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    
    while (( running_main_procs >= max_parallel )); do
        echo "[MainScript] Max parallel processes ($max_parallel) reached. Waiting for a slot..."
        
        # Check if any process has finished
        finished_pid=0
        while true; do
            for ((j=0; j<${#job_pids[@]}; j++)); do
                pid_to_check="${job_pids[$j]}"
                if ! ps -p "$pid_to_check" > /dev/null; then
                    finished_pid=$pid_to_check
                    job_pids=("${job_pids[@]:0:$j}" "${job_pids[@]:$(($j + 1))}")
                    ((running_main_procs--))
                    echo "[MainScript] Job with PID $finished_pid finished. Slots available: $((max_parallel - running_main_procs))"
                    break 2
                fi
            done
            if (( running_main_procs < max_parallel )); then
                break
            fi
            sleep 1
        done
        if (( finished_pid == 0 )) && (( running_main_procs >= max_parallel )); then
             echo "[MainScript] Fallback: Waiting for any process to finish using wait -n..."
             wait -n
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
    for ((k=0; k<${#job_pids[@]}; k++)); do
        pid_to_check_final="${job_pids[$k]}"
        if ! ps -p "$pid_to_check_final" > /dev/null; then
            finished_pid_final=$pid_to_check_final
            job_pids=("${job_pids[@]:0:$k}" "${job_pids[@]:$(($k + 1))}")
            ((running_main_procs--))
            echo "[MainScript] Final wait: Job with PID $finished_pid_final finished. Remaining: $running_main_procs"
            break
        fi
    done
    if (( finished_pid_final == 0 )) && (( running_main_procs > 0 )); then
        sleep 1 
    elif (( running_main_procs == 0 )); then
        break
    fi
done

echo "[MainScript] All main Python scripts completed." 
main_tasks_running=false

# Wait for the system monitor to stop gracefully
echo "[MainScript] Waiting for system monitor (PID: $SYSTEM_MONITOR_PID) to stop..."
if ps -p "$SYSTEM_MONITOR_PID" > /dev/null; then
    wait "$SYSTEM_MONITOR_PID" 2>/dev/null
fi

cleanup

echo "[MainScript] All tasks finished."
echo ""
echo "=========================================="
echo "Execution Summary"
echo "=========================================="
echo "Total commands executed: ${#commands[@]}"
echo "CPU cores used: $NUM_CORES"
echo "Max parallel processes: $max_parallel"

# Check result files
echo ""
echo "Generated files:"
for cmd in "${commands[@]}"; do
    # Extract file_type and file_number from the command
    if [[ $cmd =~ --file_type\ ([^[:space:]]+) ]]; then
        file_type="${BASH_REMATCH[1]}"
    fi
    if [[ $cmd =~ --file_number\ ([0-9]+) ]]; then
        file_number="${BASH_REMATCH[1]}"
    fi
    
    output_file="best_clustering_${file_type}_${file_number}.csv"
    summary_file="clustering_summary_${file_type}_${file_number}.txt"
    
    if [ -f "$output_file" ]; then
        echo "  ✓ $output_file ($(du -h "$output_file" | cut -f1))"
    fi
    if [ -f "$summary_file" ]; then
        echo "  ✓ $summary_file"
    fi
done

exit 0
