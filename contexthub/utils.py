import random 
import numpy as np
import torch
import os 
from datetime import date, datetime
import sys

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class Logger(object):
    def __init__(self, log_path, on=False):
        self.log_path = log_path
        self.on = on
        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += "+"
        else:
            if os.path.isfile(self.log_path):
                with open(self.log_path, 'w') as f:
                    f.write("")

    def log(self, string, newline=True):
        #if self.on:
        with open(self.log_path, "a") as logf:
            today = date.today()
            today_date = today.strftime("%m/%d/%Y")
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            string = today_date + ", " + current_time + ": " + string
            logf.write(string)
            if newline:
                logf.write("\n")

        sys.stdout.write(string)
        if newline:
            sys.stdout.write("\n")
        sys.stdout.flush()