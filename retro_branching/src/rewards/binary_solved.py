class BinarySolved:
    def __init__(self, not_solved=-1, solved=0):
        '''Returns solved if step solved instance, otherwise returns not_solved.'''
        self.not_solved = not_solved
        self.solved = solved

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        if done:
            return self.solved
        else:
            return self.not_solved