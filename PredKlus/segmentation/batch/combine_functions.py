class CombineFunctions():
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, *args):
        out = args
        for f in self.functions:
            out = f(*out)
        return out