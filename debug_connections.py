#!/usr/bin/env python3
"""
Debug connection handling issue
"""

# Test the issue
input_connections = {"input": []}  # Empty list

try:
    # This is what's failing
    input_conn = input_connections.get("input", [None])[0] if input_connections.get("input") else None
    print(f"Result: {input_conn}")
except KeyError as e:
    print(f"KeyError: {e}")
except IndexError as e:
    print(f"IndexError: {e}")

# The issue is that an empty list [] is truthy, so the condition passes
# but then [0] fails on an empty list

# Correct pattern:
input_conn = input_connections.get("input", [None])[0] if input_connections.get("input", []) else None
print(f"Correct result: {input_conn}")

# Even better pattern:
conns = input_connections.get("input", [])
input_conn = conns[0] if conns else None
print(f"Better result: {input_conn}")