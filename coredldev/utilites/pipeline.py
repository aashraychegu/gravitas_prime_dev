class pipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def __call__(self, base):
        for i in self.pipeline:
            base = i(base)
        return base
