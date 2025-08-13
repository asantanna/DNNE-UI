Rules DNNE Development:
* We use FAIL-FAST principles. That means:
    * No fallbacks. If something fails unexpectedly, we don't try to work around it.
        ```
            try:
                print(self.server_ip)
            except:
                # fallback to default host ip         (BAD!!!!!)
                self.server_ip = "localhost:0.0.0.0"
        ```
    * No defaults when expected objects are not found. Examples:

        ```
            x = obj.foo                     (GOOD)
            x = getattr(obj, "foo", "baz")  (BAD)
        ```    
* **Don't guess - instrument and compare**: Always add debug prints to understand behavior
* **Test incrementally**: Fix one issue at a time and verify
* **Document discoveries**: Update guides with new insights

## Development Discipline Rules

### **Root Principle**: Move methodically, not quickly. It's faster to do it right than to debug mysterious failures.

### 1. **Test Before Advancing**
* After EVERY functional change, run the code and verify it works
* Avoid batching multiple changes before testing
* If you can't run it directly, add a test script that exercises the code path

### 2. **Add Assertions and Explicit Checks**
* Use assertions to validate assumptions:
    ```python
    assert client_id is not None, "client_id must not be None"
    assert hasattr(self.workflows[workflow_id], 'client_id'), "workflow missing client_id"
    ```
* Make requirements explicit, not implicit

### 3. **Never Catch Broad Exceptions During Development**
* Optionally, remove try/except blocks when debugging
* Let things CRASH so you see the actual errors
* Only add error handling after the happy path works

### 4. **When Debugging, add Debug output FIRST, not Last**
* See the actual data flow before assuming something works
* Use print() liberally - it's better than guessing
* **Instead of trying to infer or assume things, put prints in several strategic places to find out directly!**
    * Print at entry points: `print(f"[FUNCTION_NAME] Called with: {args}")`
    * Print data transformations: `print(f"[FUNCTION_NAME] Before: {data}")`
    * Print at decision points: `print(f"[FUNCTION_NAME] Condition X={x}, taking branch Y")`
    * Print at exit points: `print(f"[FUNCTION_NAME] Returning: {result}")`
* Strategic prints solve problems in minutes that guessing would take hours to debug

### 5. **Read Existing Code Patterns**
* Check how imports are done elsewhere in the file
* Check how similar features are implemented
* Follow existing patterns, don't invent new ones

### 6. **One Change, One Test**
* Make ONE change
* Test it
* Verify it works
* Only then move to the next change

### 7. **When Debugging, Write a Simple Test Script To Isolate Code **
* Some problems are harder to debug when testing the entire server logic at once:
* Create a simple script that reproduces the issue
* Run it and see it fail
* Fix it
* Run it and see it pass
