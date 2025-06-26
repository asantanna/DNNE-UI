
# Backwards compatibility stub for CONTEXT
class ContextStub:
    """Stub node for backwards compatibility with saved workflows containing context connections"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"context": ("CONTEXT",)}}
    
    RETURN_TYPES = ("CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "passthrough"
    CATEGORY = "hidden"
    
    def passthrough(self, context=None):
        return (context,)

# Add to NODE_CLASS_MAPPINGS if it exists
if 'NODE_CLASS_MAPPINGS' in globals():
    NODE_CLASS_MAPPINGS["ContextStub"] = ContextStub
