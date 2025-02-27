import json
import os
import time
from pathlib import Path



class Logger():
    def __init__(self, config, logdir='logs', logfile='logs.json', exp_key=None, args=None):
        self.config = config
        self.logdir = Path(logdir)
        if not self.logdir.exists():
            self.logdir.mkdir(parents=True)
        self.logfile = logfile
        self.logpath = self.logdir / logfile
        if not self.logpath.exists():
            self.logpath.touch()
            self.logpath.write_text("{}")
        
        self.loss_fn = args.loss_fn
        self.dataset = args.dataset
        self.exp_time = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()) if exp_key is None else exp_key

    
    def update_logs(self, update : dict = {}, logfile=None):
        # if logfile is None:
        #     logpath = self.logpath
        # else:
        #     logpath = os.path.join(self.logdir, logfile)
        logs = json.load(self.logpath.open())
        key = self.exp_time
        if key not in logs.keys():
            logs[key] = {"dataset": self.dataset, "loss function": self.loss_fn, "config": self.config}
        for k, v in update.items():
            if k not in logs[key].keys():
                logs[key][k] = v
            elif type(logs[key][k]) == list:
                logs[key][k].append(v)
            else:
                logs[key][k] = [logs[key][k], v]
        json.dump(logs, self.logpath.open(mode="w"), indent=4)

    def save_exp(self, exec_time, acc, logfile="exp_logs.json"):
        self.update_logs(update={
          "accuracy": acc,
          "execution_time": exec_time,
        }, logfile=logfile)

    def save_train_loss(self, logfile, loss):
        self.update_logs(update={
          "losses": loss
        }, logfile=logfile)