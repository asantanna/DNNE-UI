# Debug Strategy for DNNE PPO

We are currently in the process of duplicating the IsaacGymEnvs cartpole
example inside DNNE.  We want the two PPO implementations to perform identically
in terms of the computations performed and the order of operations. We have
had difficulty achieving this (the DNNE version is not learning properly).

DNNE workflow: Cartpole_PPO (in /mnt/e/ALS-Projects/DNNE/DNNE-UI)
IsaacGymEnvs example: Cartpole (in ~/DNNE-LINUX-SUPPORT)

# Approach

The main idea is that since the DNNE implementation was derived from the
implementation used by Isaac Gym (uses rlgames, etc), then if we temporarily
disable all sources of pseudo-randomness and provide the same inputs and
environment observations, both implementations should do identical computations 
step by step. When we see a divergence, we know where the DNNE computations
are going wrong.

## Conceptual Blocks

Analyze the Isaac Gym PPO algorithm and divide it into conceptual sequential
blocks representing the different "components" of the algorithm and their sequence.
These are the categories we want in our logging. <br><br>
As a hypothecial example, it might be something like below but create your own version:
```
1. Global initialization
2. Global PPO algorithm initialization (if any)
3. Per episode initialization 
   3.1 reset env
   3.2 prepare PPO for episode
   3.x etc
4. Learning Cycle
   4.1 get environment observation
   4.2 do PPO component tasks (fwd pass, rollouts, etc)
   4.3 send action to env and step
   4.4 do PPO update
   4.x etc.
```

## Modifications to DNNE runner.py and IsaacGymEnv train.py (?)
* Create switch "--fixed-seed N" to make sure both implementations have deterministic execution
* Create switch "--debug-cycle-stop N" to stop after a certain number of PPO cycles with a hard stop such as sys.exit()
* Create switch "--debug-ppo" to enable special PPO logging.<br>Instrument both DNNE Cartpole_PPO and IsaacGymEnvs cartpole to log each phase of the algorithm as it is entered/exitted.<br>Log the results of all relevant computations.<br>Log all significant events (success, fail, episode complete, etc.)

## Debug Process
* Run claude_scripts/profiling/performance_profiler.py with --fixed-seed, --debug-ppo and --debug-cycle-stop 1 switches.
* This should generate PPO logs as described above and stop after a SINGLE PPO cycle for both DNNE and IsaacGym. We should look for any divergences. If any found, we need to identify the cause and fix. Then run again until a single cycle produces identical results.
* After a single PPO cycle produces identical results, run with ever larger values in --debug-cycle-stop

## Success Criteria
* Our logs show identical order of operations and numbers.
* We run "./dnne-test performance" and we get similar numbers for learning success
