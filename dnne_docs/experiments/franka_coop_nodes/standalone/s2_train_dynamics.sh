#!/bin/bash
# Train Dynamics network using pre-trained FK network
# This learns how actions affect the end-effector position changes

echo "==================================================="
echo "Step 2: Training Dynamics Network"
echo "==================================================="

# Activate conda environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# TIMEOUT=60 # seconds
TIMEOUT=600 # seconds

# Train dynamics network
python shadow_train_standalone.py \
  --mode train-dynamics \
  --fk-model fk_model.pt \
  --dynamics-model dynamics_model.pt \
  --timeout $TIMEOUT \
  --lr 0.001

echo ""
echo "Dynamics training complete!"
echo "Optional: Run ./s3_train_joint.sh to fine-tune both networks together"
echo "Or run ./test_models.sh to test the trained models"