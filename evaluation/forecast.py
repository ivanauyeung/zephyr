import argparse
import sys
import os
scripts_path = '/home/disk/brume/adod/zephyr/scripts'
sys.path.append('/home/disk/brume/adod/zephyr')
# Now you should be able to import 'forecast' from 'scripts'
try:
    from scripts.forecast import inference
except ImportError:
    raise ImportError("Failed to import 'forecast' from 'scripts'.")


class param_dict():
    def __init__(self, param_dict):
        os.chdir('/home/disk/brume/adod/zephyr')
        param_dict['hydra_path'] = os.path.relpath(param_dict['model_path'], os.path.join(os.getcwd(), 'hydra'))
        self.__dict__= param_dict

params = param_dict({
    'model_path' : '/home/disk/brume/adod/zephyr/outputs/hpx64_coupled-dlwp_seed0', # path to model directory 
    'model_checkpoint' : None, # none defaults to choosing the best checkpoint
    'non_strict' : True, # keep true
    'lead_time' : 336, # lead time in hours
    'forecast_init_start': '2017-01-02',
    'forecast_init_end': '2018-12-30',
    'freq' : 'biweekly',
    'batch_size' : None,
    'output_directory' : '/home/disk/brume/adod/forecasts/hpx64_coupled-dlwp_seed0/', # where  to save the forecast
    'output_filename' : '209_forecasts_hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300', # filename for the forecast
    'encode_int' : False,
    'to_zarr' : False,
    'data_directory' : '/home/quicksilver2/karlbam/Data/DLWP/HPX64', # location of data to use for the forecast
    'data_prefix' : None,
    'data_suffix' : None,
    'gpu' : 0,
})

result = inference(params)