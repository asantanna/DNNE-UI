#!/bin/bash
# Collect diverse joint-EEF pairs for FK training
# This generates random joint configurations and records the resulting EEF positions

echo "==================================================="
echo "Step 0: Collecting FK Training Data"
echo "==================================================="

# Activate conda environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Run data collection
python shadow_train_standalone.py \
  --mode collect \
  --collect-samples 10000 \
  --data-file collected_fk_data.npz

echo ""
echo "Data collection complete!"
echo "Next step: Run ./s1_train_fk_only.sh to train the FK network"