from pytorch_lightning.profiler.profiler import BaseProfiler
from collections import defaultdict
import time
import numpy as np
import pandas as pd

class CSVProfiler(BaseProfiler):

    def __init__(self,output_path=None,verbose=True):
        """
        :param output_path (str): The path where the profiler will save the resulting .CSV. (Optional)
        :param verbose (bool): Print the profiler results on screen, using print() 
        """
        self.output_path = output_path
        self.verbose = verbose
        self.current_actions = {}
        self.recorded_durations = defaultdict(list)

    def start(self, action_name):
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name):
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def describe(self):
        log = []    
        for action, durations in self.recorded_durations.items():
            log.append([action,np.mean(durations),np.sum(durations)])
        log = np.array(log)
        log_df = pd.DataFrame(columns=['mean_duration', 'total_time'],index=log[:,0],data=log[:,1:])
        log_df = log_df.astype('float32')
        if self.verbose:
            print("\033[1mProfiler Report\033[0m")            
            floatformat = '{:,.2f}'.format            
            with pd.option_context('display.max_rows', None, 'display.float_format',floatformat): 
                print(log_df)
        if len(self.output_path)>0:
            log_df.to_csv(self.output_path)
            if self.verbose:
                print(f"\nProfiler output saved to: {self.output_path}")
