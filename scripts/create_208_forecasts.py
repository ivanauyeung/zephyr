import argparse
import sys
import os
sys.path.append('/home/disk/brume/adod/zephyr')
from scripts.forecast import inference


class param_dict():
    def __init__(self, param_dict):
        os.chdir('/home/disk/brume/adod/zephyr')
        param_dict['hydra_path'] = os.path.relpath(param_dict['model_path'], os.path.join(os.getcwd(), 'hydra'))
        self.__dict__= param_dict

params = param_dict({
    'model_path' : '/home/disk/brume/adod/zephyr/outputs/hpx64_gru_1x10', # path to model directory 
    'model_checkpoint' : None, # none defaults to choosing the best checkpoint
    'non_strict' : True, # keep true
    'lead_time' : 336, # lead time in hours
    'forecast_init_start': '2017-01-02',
    'forecast_init_end': '2018-12-30',
    'freq' : 'biweekly',
    'batch_size' : None,
    'output_directory' : '/home/disk/brume/adod/forecasts/hpx64_gru_1x10/', # where  to save the forecast
    'output_filename' : '208_forecasts_hpx64_unet_136-68-34_cnxt_1x1_gru_6h_300', # filename for the forecast
    'encode_int' : False,
    'to_zarr' : False,
    'data_directory' : '/home/disk/quicksilver2/karlbam/Data/DLWP/HPX64', # location of data to use for the forecast
    'data_prefix' : None,
    'data_suffix' : None,
    'gpu' : 0,
})

result = inference(params)