class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.best_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 Early stopping triggered after {self.counter} epochs without improvement.")
        
        return self.early_stop
    
'''
example

def train():
    early_stopper = EarlyStopper(patience=15, min_delta=0.001)

    for epoch in epochs:
    
        ////////(val_loss까지 계산)//////////

        should_stop = early_stopper(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

'''