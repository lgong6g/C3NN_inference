import optuna
from main import *
from optuna.visualization import plot_optimization_history, plot_contour, plot_param_importances    # it needs plotly and kaleido
from hyperparameters import hparams


# Objective function to minimize
def objective(trial):

    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("flow_learning_rate", 1e-6, 1e-3, log=True)
    #weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-6, log=True)
    #weight_decay = 1.e-7
    n_layers = trial.suggest_int("n_layers", 1, 5)
    flow_num_transforms = trial.suggest_int("flow_num_transforms", 2, 8)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    flow_hidden_features = trial.suggest_categorical("flow_hidden_features", [32, 64, 128, 256, 512])
    r_link = trial.suggest_float("r_link", 5.e-3, 5.e-2, log=True)
    flow_training_batch_size = trial.suggest_categorical("flow_training_batch_size", [5, 10, 20, 30, 40, 50])
    n_dims_summ = trial.suggest_int("n_dims_summ", 2, 200)

    # Some verbose
    print('\nTrial number: {}'.format(trial.number))
    print('flow_learning_rate: {}'.format(flow_learning_rate))
    #print('weight_decay: {}'.format(weight_decay))
    print('n_layers:  {}'.format(n_layers))
    print('flow_num_transforms:  {}'.format(flow_num_transforms))
    print('hidden_channels:  {}'.format(hidden_channels))
    print('flow_hidden_features:  {}'.format(flow_hidden_features))
    print('r_link:  {}'.format(r_link))
    print('flow_training_batch_size:  {}'.format(flow_training_batch_size))
    print('n_dims_summ:  {}'.format(n_dims_summ))

    # Hyperparameters to be optimized
    hparams.flow_learning_rate = flow_learning_rate
    #hparams.weight_decay = weight_decay
    hparams.n_layers = n_layers
    hparams.flow_num_transforms = flow_num_transforms
    hparams.hidden_channels = hidden_channels
    hparams.flow_hidden_features = flow_hidden_features
    hparams.r_link = r_link
    hparams.flow_training_batch_size = flow_training_batch_size
    hparams.n_dims_summ = n_dims_summ

    # Default params
    #hparams.n_epochs = n_epochs
    hparams.simsuite = simsuite
    hparams.snap = snap
    hparams.simset = simset
    hparams.n_sims = n_sims

    # Run main routine
    val_prob = main(hparams, verbose = True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_prob


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Optuna parameters
    storage = "sqlite:///"+os.getcwd()+"/optuna_"+simsuite+"_"+simset+'_newrun2'#+'_onlypos'#+"_nbody"
    study_name = "joint_8_hparams_ISA_newrun2"
    n_trials   = 1000

    # Define sampler and start optimization
    sampler = optuna.samplers.TPESampler(n_startup_trials=50)#10)
    study = optuna.create_study(direction='maximize', study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials, gc_after_trial=True)

    # Print info for best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    hparams.flow_learning_rate = trial.params['flow_learning_rate']
    #hparams.weight_decay = weight_decay
    hparams.n_layers = trial.params["n_layers"] 
    hparams.flow_num_transforms = trial.params['flow_num_transforms']
    hparams.hidden_channels = trial.params['hidden_channels']
    hparams.flow_hidden_features = trial.params['flow_hidden_features']
    hparams.r_link = trial.params['r_link']
    hparams.flow_training_batch_size = trial.params['flow_training_batch_size']
    hparams.n_dims_summ = trial.params['n_dims_summ']

    # Save best model and plots
    if not os.path.exists("Best"):
        os.mkdir("Best")
    # Change nominal suite to read correct files (actually in ps mode both suites are employed)
    if hparams.outmode=="ps":
        hparams.simsuite = hparams.flip_suite()
    files = []
    #files.append( "Plots/out_true_Om_"+hparams.name_model()+".png" )
    #files.append( "Plots/out_true_Sig_"+hparams.name_model()+".png" )
    #files.append( "Plots/loss_"+hparams.name_model()+".png" )
    #iles.append( "Plots/ps_"+hparams.name_model()+".png" )
    #iles.append( "Plots/rel_err_"+hparams.name_model()+".png" )
    files.append( "Models/joint_training/"+hparams.name_model()+'_inference.pkl')
    files.append( "Models/joint_training/"+hparams.name_model()+'_density_estimator.pkl')
    files.append( "Plots/joint_training/posterior_"+hparams.name_model()+'.png' )
    for file in files:
        if os.path.exists(file):
            os.system("cp "+file+" Best/.")

    # Visualization of optimization results
    fig = plot_optimization_history(study)
    fig.write_image("Plots/newrun2/optuna_optimization_history_"+simsuite+".png")

    fig = plot_contour(study)#, params=["learning_rate", "weight_decay", "r_link"])#, "use_model"])
    fig.update_layout(font=dict(size=12),width=900,height=900)
    fig.write_image("Plots/newrun2/optuna_contour_"+simsuite+".png")

    fig = plot_param_importances(study)
    fig.write_image("Plots/newrun2/plot_param_importances_"+simsuite+".png")

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
