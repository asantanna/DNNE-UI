#!/usr/bin/env python3
"""
Compare PPO cycle outputs between DNNE and IGE
"""
import re
import sys

def parse_debug_output(filename):
    """Extract key PPO cycle debug values from log file"""
    data = {
        'steps': [],
        'actions': [],
        'values': [],
        'rewards': [],
        'advantages': [],
        'returns': [],
        'loss': None,
        'parameters': {}
    }
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract hyperparameters if shown
    param_patterns = {
        'horizon_length': r'horizon[_\s]+length[:\s]+(\d+)',
        'mini_epochs': r'mini[_\s]+epochs[:\s]+(\d+)',
        'minibatch_size': r'minibatch[_\s]+size[:\s]+(\d+)',
        'num_envs': r'num[_\s]+envs[:\s]+(\d+)',
        'gamma': r'gamma[:\s]+([\d.]+)',
        'tau': r'tau[:\s]+([\d.]+)'
    }
    
    for param, pattern in param_patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            data['parameters'][param] = match.group(1)
    
    # Extract PPO_CYCLE debug lines
    cycle_pattern = r'\[(?:DNNE_DEBUG|IGE)\] PPO_CYCLE: Step (\d+): action=([-\d.]+), value=([-\d.]+), reward=([-\d.]+)'
    for match in re.finditer(cycle_pattern, content):
        step, action, value, reward = match.groups()
        data['steps'].append(int(step))
        data['actions'].append(float(action))
        data['values'].append(float(value))
        data['rewards'].append(float(reward))
    
    # Extract PPO_BATCH debug info
    batch_pattern = r'PPO_BATCH:.*?Advantages.*?mean: ([-\d.]+).*?std: ([-\d.]+)'
    adv_match = re.search(batch_pattern, content)
    if adv_match:
        data['adv_mean'] = float(adv_match.group(1))
        data['adv_std'] = float(adv_match.group(2))
    
    # Extract loss
    loss_pattern = r'loss: ([-\d.]+)'
    loss_match = re.search(loss_pattern, content)
    if loss_match:
        data['loss'] = float(loss_match.group(1))
    
    return data

def compare_outputs():
    """Compare DNNE and IGE outputs"""
    dnne_data = parse_debug_output('/tmp/dnne_ppo_cycle_output.log')
    ige_data = parse_debug_output('/tmp/ige_ppo_cycle_output.log')
    
    print("=== PPO Cycle Comparison: DNNE vs IGE ===\n")
    
    # Show detected parameters
    print("Detected Parameters:")
    params_to_show = ['horizon_length', 'mini_epochs', 'minibatch_size', 'num_envs', 'gamma', 'tau']
    for param in params_to_show:
        dnne_val = dnne_data['parameters'].get(param, 'not found')
        ige_val = ige_data['parameters'].get(param, 'not found')
        print(f"  {param}: DNNE={dnne_val}, IGE={ige_val}")
    
    # Compare steps
    print(f"\nSteps collected: DNNE={len(dnne_data['steps'])}, IGE={len(ige_data['steps'])}")
    
    # Compare first few values
    if dnne_data['actions'] and ige_data['actions']:
        print("\nFirst 5 actions:")
        print(f"DNNE: {dnne_data['actions'][:5]}")
        print(f"IGE:  {ige_data['actions'][:5]}")
        
        # Check for differences
        if len(dnne_data['actions']) >= 5 and len(ige_data['actions']) >= 5:
            max_diff = max(abs(d - i) for d, i in zip(dnne_data['actions'][:5], ige_data['actions'][:5]))
            print(f"Max difference: {max_diff:.6f}")
        
    if dnne_data['values'] and ige_data['values']:
        print("\nFirst 5 values:")
        print(f"DNNE: {dnne_data['values'][:5]}")
        print(f"IGE:  {ige_data['values'][:5]}")
        
        # Check for differences
        if len(dnne_data['values']) >= 5 and len(ige_data['values']) >= 5:
            max_diff = max(abs(d - i) for d, i in zip(dnne_data['values'][:5], ige_data['values'][:5]))
            print(f"Max difference: {max_diff:.6f}")
    
    # Compare advantages
    if 'adv_mean' in dnne_data and 'adv_mean' in ige_data:
        print(f"\nAdvantages mean: DNNE={dnne_data['adv_mean']:.4f}, IGE={ige_data['adv_mean']:.4f}")
        print(f"Advantages std:  DNNE={dnne_data['adv_std']:.4f}, IGE={ige_data['adv_std']:.4f}")
        
        mean_diff = abs(dnne_data['adv_mean'] - ige_data['adv_mean'])
        std_diff = abs(dnne_data['adv_std'] - ige_data['adv_std'])
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Std difference: {std_diff:.6f}")
    
    # Compare loss
    if dnne_data['loss'] and ige_data['loss']:
        print(f"\nPPO Loss: DNNE={dnne_data['loss']:.4f}, IGE={ige_data['loss']:.4f}")
        diff = abs(dnne_data['loss'] - ige_data['loss'])
        print(f"Loss difference: {diff:.6f} ({diff/max(dnne_data['loss'], ige_data['loss'])*100:.2f}%)")

if __name__ == '__main__':
    compare_outputs()