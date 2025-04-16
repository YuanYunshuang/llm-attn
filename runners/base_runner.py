

from src.utils.train_utils import *
from src.runners.hooks import Hooks


class BaseRunner:
    def __init__(self,
                 dataloader,
                 model,
                 gpus=0,
                 log_every=10,
                 hooks=None,
                 **kwargs
                 ):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.total_iter = len(dataloader)
        self.iter = 1
        self.epoch = 1

        self.model = model
        self.hooks = Hooks(hooks)

        self.gpus = gpus
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.log_every = log_every

        self.init()

    def init(self):
        if self.model is not None:
            self.model.to(self.device)

    def setup_logger(self, *args, **kwargs):
        pass

    def set_logdir(self, logdir):
        self.logger.log_path = logdir

    @property
    def logdir(self):
        if hasattr(self, 'logger'):
            return self.logger.logdir
        else:
            return None

    def run(self):
        raise NotImplementedError

    def next_batch(self):
        if self.iter >= self.total_iter:
            self.iter = 1
            self.epoch += 1
            self.data_iter = iter(self.dataloader)
        batch = next(self.data_iter)
        return batch





