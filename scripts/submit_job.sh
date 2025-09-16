#!/bin/bash

# Simple wrapper to submit PBS jobs
# Usage: ./submit_job.sh "your command here"

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"command to run\""
    echo "Example: $0 \"uv run sft @ configs/debug/sft.toml\""
    exit 1
fi

# Create a temporary job script
TMPFILE=$(mktemp /tmp/pbs_job.XXXXXX)
cat > "$TMPFILE" << EOF
#!/bin/tcsh
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=120gb
#PBS -l walltime=03:00:00
#PBS -N prime
#PBS -M pxm5426@psu.edu
#PBS -m bea
#PBS -o pbs_results/
#PBS -e pbs_results/

source /scratch/pxm5426/.tcshrc
cd /scratch/pxm5426/repos/prime-rl

# Run the command
$1
EOF

# Submit the job
qsub "$TMPFILE"

# Clean up
rm "$TMPFILE" 