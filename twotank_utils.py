from neuromancer.loggers import BasicLogger
from neuromancer.callbacks import Callback
import torch
import copy

class CallbackChild(Callback):
    def begin_train(self,trainer):
        trainer.train_losses_epoch = []
        trainer.dev_losses_epoch   = []
        trainer.dev_xresids_epoch   = []
        trainer.dev_uresids_epoch   = []

        trainer.model_list = []
        # save the model before any training
        trainer.model_list.append( copy.deepcopy(trainer.model) )


    def end_epoch(self,trainer,output):

        trainer.dev_losses_epoch.append( output['dev_loss'].item() )

        if (trainer.current_epoch < 10) or (trainer.current_epoch==trainer.epochs-1):
            trainer.model_list.append( copy.deepcopy(trainer.model) )
