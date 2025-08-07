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

### 1. **Test Every Single Change**
* After EVERY edit, run the code and verify it works
* No batching multiple changes before testing
* If you can't run it directly, add a test script that exercises the code path

### 2. **Add Assertions and Explicit Checks**
* Use assertions to validate assumptions:
    ```python
    assert client_id is not None, "client_id must not be None"
    assert hasattr(self.workflows[workflow_id], 'client_id'), "workflow missing client_id"
    ```
* Make requirements explicit, not implicit

### 3. **Never Catch Broad Exceptions During Development**
* Remove try/except blocks when debugging
* Let things CRASH so you see the actual errors
* Only add error handling after the happy path works

### 4. **Add Debug Output FIRST, Not Last**
* Before making ANY functional change, add logging
* See the actual data flow before assuming anything
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

### 7. **Write a Failing Test First**
* Create a simple script that reproduces the issue
* Run it and see it fail
* Fix it
* Run it and see it pass

### 8. **Verify Your Own Code**
* After adding code that uses a module, grep to verify it's imported
* After adding a function call, verify the function exists
* After accessing an attribute, verify the object has that attribute

**Root Principle**: Move methodically, not quickly. It's faster to do it right than to debug mysterious failures.
