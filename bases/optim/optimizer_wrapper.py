import inspect

class OptimizerWrapper:
    """
    A wrapper to make optimizer more concise
    """

    def __init__(self, model, optimizer, lr_scheduler=None,):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


    def step(self, inputs, labels):
        self.zero_grad()
        loss = self.model.loss(inputs, labels)
        #print('optimizer  :'+str(loss))

        loss.backward()
        #print('grade   ' + str(self.model.features[0].weight.grad.sum()))
        return self.optimizer.step(), loss

    def zero_grad(self):
        self.model.zero_grad()

    def lr_scheduler_step(self, loss=None):#but self.lr_scheduler is None
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            # if 'metrics' in inspect.signature(self.lr_scheduler.step).parameters.keys():
            #     self.lr_scheduler.step(loss)
            # else:



    def get_last_lr(self):
        if self.lr_scheduler is None:
            return self.optimizer.defaults["lr"]
        else:
            return self.optimizer.param_groups[0]['lr']
