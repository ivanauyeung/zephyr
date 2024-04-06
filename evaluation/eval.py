import copy
import pprint
import sys 
import xarray as xr 
sys.path.append('/home/disk/brume/adod/zephyr')
import matplotlib.pyplot as plt
import evaluation.evaluators as ev

# Compares RMSE from multiple models over multiple vairables 

# list of variables to evaluate
eval_vars = [ 'z500',]

# variables names as they appear in the raw era5 data -- used for verification 
era5_varname = [ 'geopotential_500',]

# a template that identifies the forecast specific parameters ( file, plot_kwargs, label, climatology) and 
# parameters that should be consistent accross forecasts (verification_path, eval_variable, plot_file)
PARAM_template = {
    'forecast_params' : [
        {'file':'/home/disk/brume/adod/forecasts/hpx64_lstm_1x10/208_forecasts_hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300.nc',
         'plot_kwargs':{'color' : 'pink','linewidth' : '.75'},
         'label':'LSTM Model',
         'climatology' : '/home/disk/quicksilver/nacc/S2S/z500_climo_test.nc',
        },
        {'file':'home/disk/brume/adod/WeeklyNotebooks/2023.12.25/NewBaseline/new_baseline_180-90-90_best.nc',
         'plot_kwargs':{'color' : 'orange','linewidth' : '.75'},
         'label':'Baseline - best (300)',
         'climatology' : '/home/disk/quicksilver/nacc/S2S/z500_climo_test.nc',
        }
    ],
    'verification_path' : '/home/disk/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_geopotential_500.nc',
    'eval_variable' : 'z500',
    'plot_file' : '/home/disk/brume/adod/WeeklyNotebooks/2024.3.28/NewBaseline/evaluate_new_baseline_180-90-90.pdf',
    # limit of x axis in days. supports comparison of forecasts with different lead times 
#    'xlim':None,
    'xlim':{'left':0,'right':14},
}

def main(params):

    # create evauator objects 
    evaluators = []
    for fcst in params['forecast_params']:
        # initialize evaluator for forecast
        evaluators.append(ev.EvaluatorHPX(
            forecast_path = fcst['file'],
            verification_path = params['verification_path'],
            eval_variable = params['eval_variable'],
            remap_config = None,
            on_latlon = True,
            times = "2017-01-01--2018-12-31",
            #times = "2017-01-01--2017-01-03",
            poolsize = 30,
            verbose = True,
            ll_file=f'{fcst["file"][:-3]}_{params["eval_variable"]}_ll.nc',
            )
        )
    # generate or inheret verification 
    for i,fcst in enumerate(params['forecast_params']):
        if i==0:
            evaluators[i].generate_verification(verification_path=params['verification_path'])
        else:
            for j in range(0,i):
                needs_verif = True
                if evaluators[i].forecast_da.shape == evaluators[j].forecast_da.shape and needs_verif:
                    evaluators[i].set_verification(evaluators[j].verification_da)
                    needs_verif = False #once verification is inhereted we no longer need to assign one 
            else:
                print(f'unable to inherit verification from {evaluators[0].forecast_path} to {evaluators[i].forecast_path}. Generating new verification.')
                evaluators[i].generate_verification(verification_path=params['verification_path'])
    # calculate acc and rmse 
    accs = []
    rmses = []
    for i, e in enumerate(evaluators):
        accs.append(
            e.get_acc(
                climatology_path = params['forecast_params'][i]['climatology'],
                mean = False
            )
        )
        rmses.append(
            e.get_rmse(
                mean = False,
            )
        )
    
    # plot metrics 
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
    for a in axs: a.grid() 
    for i,fcst in enumerate(params['forecast_params']):
        # plot acc 
        evaluators[i].plot_acc(
            accs[i],
            ax=axs[1],
            kwargs=fcst['plot_kwargs'],
        )
        # plot rmse 
        evaluators[i].plot_rmse(
            rmses[i],
            ax=axs[0],
            kwargs=fcst['plot_kwargs'],
            model_name=fcst['label'],
        )
    # formatting and labels 
    label = evaluators[-1].variable_metas[params['eval_variable']]['plot_label']
    unit = evaluators[-1].variable_metas[params['eval_variable']]['unit']
    axs[0].set_ylabel(f"{label} RMSE ({unit})",fontsize=15)
    axs[1].set_ylabel(f"{label} ACC",fontsize=15)
    axs[0].set_xlabel('forecast day',fontsize=15)
    axs[1].set_xlabel('forecast day',fontsize=15)
    axs[0].legend(fontsize=12, loc=4)
    if params['xlim'] is not None:
        axs[0].set_xlim(**params['xlim'])
        axs[1].set_xlim(**params['xlim'])
    fig.tight_layout()
    # save 
    plt.savefig(params['plot_file'])

if __name__=="__main__":
    
    # uses param template to populate a parameter for each variable
    param_list = []
    for i,var in enumerate(eval_vars):
        param_list.append(copy.deepcopy(PARAM_template))
        for d in param_list[-1]['forecast_params']:
            d['climatology'] = d['climatology'].replace('z500',var)
            if 'z500' in d['file']:
                d['file'] = d['file'].replace('z500',var)
        param_list[-1]['eval_variable'] = var
        param_list[-1]['plot_file'] = param_list[-1]['plot_file'].replace('z500',var)
        param_list[-1]['verification_path'] = param_list[-1]['verification_path'].replace('geopotential_500', era5_varname[i])

    for p in param_list:
        main(p)