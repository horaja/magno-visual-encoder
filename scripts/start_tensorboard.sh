#!/bin/bash
source configs/default_config.sh

echo "Starting TensorBoard..."
echo "Logs directory: ${TENSORBOARD_LOG_DIR}"
echo "Open http://localhost:6006 in your browser"

ssh -L 6006:localhost:6006 horaja@$HOSTNAME

tensorboard --logdir ${TENSORBOARD_LOG_DIR} --port 6006 --reload_interval 30