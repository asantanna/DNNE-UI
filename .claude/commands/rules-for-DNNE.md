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
