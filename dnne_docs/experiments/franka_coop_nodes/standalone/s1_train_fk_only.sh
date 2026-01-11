#!/bin/bash
# Train Forward Kinematics network on collected data
# This learns the mapping from joint angles to end-effector positions

echo "==================================================="
echo "Step 1: Training Forward Kinematics Network"
echo "==================================================="

# Activate conda environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# NUM_EPOCHS=500
NUM_EPOCHS=3000

# Train FK network
python shadow_train_standalone.py \
  --mode train-fk \
  --data-file collected_fk_data.npz \
  --fk-model fk_model.pt \
  --fk-epochs $NUM_EPOCHS \
  --batch-size 32 \
  --lr 0.001 \
  --show-progress

echo ""
echo "FK training complete!"
echo "Next step: Run ./s2_train_dynamics.sh to train the dynamics network"