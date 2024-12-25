#!/bin/bash
SCRIPT=$1   # The target script for run.

WORK_DIR=$HOME/workspace/epymarl_based  # Path to root dir.

if ! [ -d "$WORK_DIR" ]; then
    echo "Wrong work dir: $WORK_DIR"
    exit 1
fi

LOG_DIR=$WORK_DIR/log
LOGFILE=$LOG_DIR/terminal_log.log

mkdir -p "$LOG_DIR"

if [[ $SCRIPT = /* ]]; then
    SCRIPT_PATH=$SCRIPT
else
    SCRIPT_PATH=$WORK_DIR/$SCRIPT # Path to train script.
    if ! [[ -e $SCRIPT_PATH ]]; then
        echo "$(timestamp) | FATAL    | bash         | Run Failed, $SCRIPT_PATH not found." | tee -a "$LOGFILE"
    fi
fi

timestamp() { 
    date +"%Y-%m-%d_%H-%M-%S" 
}

SEPERATOR="========================================================================================================================"

if ! nohup "$SCRIPT_PATH" </dev/null &> >(tee -a "$LOGFILE"); then
    echo "$(timestamp) | FATAL    | bash         | Run Failed, see log files in $LOG_DIR for more information." | tee -a "$LOGFILE"
    echo "$(timestamp) | FATAL    | bash         | Train script ends." | tee -a "$LOGFILE"
    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$LOGFILE"
    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$LOGFILE"
    exit 1
fi
