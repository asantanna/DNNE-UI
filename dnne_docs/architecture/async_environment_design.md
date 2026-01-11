# DNNE Async Environment Design

*See also: [Queue Framework](queue_framework.md), [Adaptive Yielding](adaptive_yielding.md)*

## Overview

This document explains the architectural decisions behind DNNE's integration with Isaac Gym environments, particularly the async design choices and minimal intervention approach.

## Key Architectural Constraint

**DNNE must support concurrent async networks** running in the same workflow. This is fundamental to DNNE's vision of multi-modal AI systems where:

- **Vision networks** process visual inputs
- **Hearing networks** process audio
- **Balance/proprioception networks** handle body state
- **Decision networks** coordinate actions

All these networks must run **concurrently without blocking each other**.

## Design Principles

### 1. Preserve Async Architecture
- Never block other nodes
- Maintain queue-based communication
- Support concurrent network execution

## Why This Architecture Matters

DNNE's vision is to enable complex, multi-modal AI systems that mirror biological intelligence:

- **Parallel Processing**: Like the brain, different networks process different modalities simultaneously
- **Non-blocking Execution**: One slow network doesn't freeze the entire system
- **Scalability**: Easy to add new sensory networks without redesigning the system
- **Real-time Performance**: Critical for robotics applications

The async queue-based architecture is not just a technical choice - it's fundamental to achieving these goals.

## References
- DNNE queue framework: `/home/asantanna/DNNE/DNNE-UI/export_system/templates/base/queue_framework.py`
