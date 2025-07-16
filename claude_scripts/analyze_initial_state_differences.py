#!/usr/bin/env python3
"""Analyze initial state differences between DNNE and IsaacGymEnvs."""

import json

def analyze_initial_states():
    """Compare initial states between DNNE and IsaacGymEnvs."""
    
    print("=" * 80)
    print("INITIAL STATE COMPARISON - DNNE vs IsaacGymEnvs")
    print("=" * 80)
    
    # DNNE initial state (from logs)
    dnne_obs = [0.0, 0.0, 0.0, 0.0]
    dnne_obs_norm = [0.0, 0.0, 0.0, 0.0]
    dnne_obs_mean = [0.0, 0.0, 0.0, 0.0]
    dnne_obs_std = [0.00045311416033655405, 0.00045311416033655405, 0.00045311416033655405, 0.00045311416033655405]
    dnne_layer1_weights = [0.3822692632675171, 0.4150039553642273, -0.11713624000549316, 0.45930564403533936]
    dnne_layer1_bias = [0.4192349314689636, -0.099232017993927, 0.4301983714103699, 0.1557910442352295]
    dnne_policy_weights = [0.1325194090604782, 0.001935910084284842, 0.08523331582546234, -0.061804939061403275]
    dnne_policy_bias = [-0.08017829060554504]
    dnne_log_std = [0.0]
    
    # IsaacGymEnvs initial state (from logs)
    isaac_obs = [0.09754373133182526, 0.16293340921401978, -0.07422007620334625, -0.14360617101192474]
    isaac_layer1_weights = [-0.10955178737640381, 0.10089534521102905, -0.24342751502990723, 0.2936413288116455]
    isaac_layer1_bias = [0.0, 0.0, 0.0, 0.0]
    isaac_policy_weights = [-0.016931548714637756, -0.052127644419670105, 0.16241896152496338, -0.0385725200176239]
    isaac_policy_bias = [0.0]
    
    # First action produced
    dnne_action1 = 0.0110
    isaac_action1 = 0.1940
    
    print("\n🔍 KEY DIFFERENCES FOUND:")
    print("-" * 80)
    
    print("\n1. INITIAL OBSERVATIONS:")
    print(f"   DNNE:        {dnne_obs}")
    print(f"   IsaacGym:    {isaac_obs}")
    print(f"   ⚠️  DNNE starts with all zeros, IsaacGym has random initial state!")
    
    print("\n2. OBSERVATION NORMALIZATION:")
    print(f"   DNNE normalized obs: {dnne_obs_norm}")
    print(f"   DNNE obs mean:       {dnne_obs_mean}")
    print(f"   DNNE obs std:        {dnne_obs_std}")
    print(f"   ⚠️  DNNE normalizes with near-zero std (epsilon), making normalized obs = 0")
    
    print("\n3. NETWORK INITIALIZATION:")
    print(f"   First layer weights (first 4):")
    print(f"     DNNE:      {dnne_layer1_weights}")
    print(f"     IsaacGym:  {isaac_layer1_weights}")
    print(f"   First layer bias (first 4):")
    print(f"     DNNE:      {dnne_layer1_bias}")
    print(f"     IsaacGym:  {isaac_layer1_bias}")
    print(f"   ⚠️  Different weight initialization schemes!")
    
    print("\n4. POLICY HEAD:")
    print(f"   Policy weights (first 4):")
    print(f"     DNNE:      {dnne_policy_weights}")
    print(f"     IsaacGym:  {isaac_policy_weights}")
    print(f"   Policy bias:")
    print(f"     DNNE:      {dnne_policy_bias}")
    print(f"     IsaacGym:  {isaac_policy_bias}")
    print(f"   Log std:")
    print(f"     DNNE:      {dnne_log_std}")
    print(f"     IsaacGym:  Not shown (might be different)")
    
    print("\n5. FIRST ACTION:")
    print(f"   DNNE:      {dnne_action1:.4f}")
    print(f"   IsaacGym:  {isaac_action1:.4f}")
    print(f"   Difference: {abs(isaac_action1 - dnne_action1):.4f}")
    
    print("\n" + "=" * 80)
    print("ROOT CAUSES IDENTIFIED:")
    print("=" * 80)
    
    print("\n🔴 CRITICAL ISSUE #1: Initial Environment State")
    print("   - DNNE starts with all observations = 0.0")
    print("   - IsaacGym starts with random non-zero observations")
    print("   - This suggests DNNE is not properly resetting the environment")
    
    print("\n🔴 CRITICAL ISSUE #2: Network Weight Initialization")
    print("   - DNNE uses PyTorch default initialization")
    print("   - IsaacGym uses different initialization (likely rl_games specific)")
    print("   - Different bias initialization: DNNE has non-zero bias, IsaacGym has zero")
    
    print("\n🔴 CRITICAL ISSUE #3: Observation Normalization")
    print("   - DNNE's RunningMeanStd starts with epsilon std")
    print("   - This causes division by near-zero, making all normalized obs = 0")
    print("   - IsaacGym likely has different normalization initialization")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("\n1. Fix environment reset to generate proper initial observations")
    print("2. Match network initialization with rl_games")
    print("3. Fix observation normalization initialization")
    print("4. Verify fixed seed is applied to environment reset")
    
if __name__ == "__main__":
    analyze_initial_states()