import argparse
import time
from typing import Tuple, Optional

from configuration import config, batch_runner
from runs import runs_manager as run_man
from src.utils.wandb_data_fetcher import download_run
from src.utils.wandb_utils import wandb_init


def load_simple_config(config_path: str, wandb_resume_run_fn, wandb_new_run_fn, ngpus: Optional[int] = None,
                       use_wandb_override=True):
    """
    Used for loading a normal run that is not part of a batch run. Therefore it is much more simple than loading a batch
    config

    Steps:
    * Read config to get run name and wandb related info
    * Read saved config if it is a saved run
    * Load wandb if wandb is requested
    * Read original config again for overwrites
    * Overwrite n_gpus option if required
    * Save config to run folder

    @param config_path: path to the config, can be relative to configuration/configs
    @param wandb_resume_run_fn: function that allows wandb to resume
    @param wandb_new_run_fn: function that creates a new wandb run
    @param ngpus: number of gpus if config option should be overridden
    :param use_wandb_override: determines whether wandb actions should be taken in the config loading
    """
    config.read(config_path)
    run_name = config.run_name
    print(f'Run name: {run_name}')

    if run_man.run_folder_exists(run_name):
        print('Run folder already exists, reading its config')
        run_man.load_config(run_name)  # load saved config
        if config.use_wandb and use_wandb_override:
            wandb_resume_run_fn()
    else:
        print(f'No runs folder detected with name {run_name}. Creating one')
        if config.use_wandb and use_wandb_override:
            wandb_new_run_fn()

        run_man.set_up_run_folder(config.run_name, use_wandb_override)

    config.read(config_path)  # overwrite saved/wandb config with provided config (only values present in this config)

    if ngpus is not None:  # n_gpu override
        config.n_gpus = ngpus

    run_man.save_config(run_name, use_wandb_override=use_wandb_override)
    print(f'config: {config.__dict__}')


def load_batch_config():
    """
    there are 3 possible levels of configs to be loaded:
    1: a saved config which is attached to an existing run which has been executed before
        this config does not exist when starting a fresh run, only when continuing an existing one
    2: a scheduled config. If a run scheduler is used, it will point to a one of the configs in its schedule
    3: the cli config, which is specified as a run arg to the main program

    when no run schedule is used, the cli config values overwrite the saved config (if one exists)
        an example of when this is desirable is to change the num gpu's when continuing a run, or
        to change the man num of generations, to evolve a population for longer

    when a run schedule is specified, it will fetch a config file eg: elite.json
    It may be desirable to override certain properties of all runs in a schedule
        An example of this is schedule = {elite,base} - we may want to turn on DataAug for bot
        ie: transform the schedule into {da_elite,da_base}

    thus when a run schedule is used, the cli config starting the schedule may contain overriding config values (eg: da=true)

    therefore the priority of configs when a schedule is being used is:
        saved config (if exists)    - lowest
        scheduled config            - middle
        cli config                  - highest
    """
    cli_args = get_cli_args()
    stagger(cli_args.stagger_number)

    effective_run_name, scheduled_cfg_file_name = get_batch_schedule_run_names(cli_args)
    effective_run_name = load_saved_config(effective_run_name, cli_args.config)

    if scheduled_cfg_file_name:
        print(f'reading scheduled config: {scheduled_cfg_file_name}')
        config.read(scheduled_cfg_file_name)

    print(f'Reading cli config {cli_args.config}')
    config.read(cli_args.config)  # final authority on config values

    # must detect whether the scheduler is calling for a fully train, or an evolutionary run
    fully_train, resume_fully_train = batch_runner.get_fully_train_state(effective_run_name)
    print(f'scheduler is starting run with FT = {fully_train} continue FT = {resume_fully_train}')
    config.fully_train = fully_train
    config.resume_fully_train = resume_fully_train

    config.run_name = effective_run_name
    if cli_args.ngpus is not None:
        config.n_gpus = cli_args.ngpus
    else:
        print(f'no gpu argument given, using config value of {config.n_gpus}')

    # Full config is now loaded
    if config.use_wandb and not fully_train:
        wandb_init()

    if not run_man.run_folder_exists(config.run_name):
        print(f'New run, setting up run folder for {config.run_name}')
        run_man.set_up_run_folder(config.run_name)

    print(f'Saving conf to run {config.run_name}')
    run_man.save_config(config.run_name)
    print(f'config: {config.__dict__}')


def get_batch_schedule_run_names(cli_args) -> Tuple[str, str]:
    """
    finds out the effective run name to be used
    if there is a run scheduler, the run config it is scheduling is fetched, but not parsed
    the effective name, as well as the Optional scheduled_run_name are returned
    """
    batch_run_scheduler = config.read_option(cli_args.config, 'batch_run_scheduler')
    cli_cfg_run_name = config.read_option(cli_args.config, 'run_name')

    if not batch_run_scheduler:
        raise Exception('Could not find batch_run_schedule option in config, but trying to read batch config')

    # there is a batch run scheduler so must use the config specified in it
    scheduled_cfg_file_name, scheduled_run_name = batch_runner.get_config_path(batch_run_scheduler, cli_cfg_run_name)
    return scheduled_run_name, scheduled_cfg_file_name


def load_saved_config(effective_run_name, cli_cfg_file_name):
    wandb_run_path = config.read_option(cli_cfg_file_name, 'wandb_run_path')

    if wandb_run_path:
        # wandb downloaded runs are not used by the batch scheduler.
        # this must be a standalone run
        print(f'downloading run from wandb with path: {wandb_run_path}')
        # if a wandb run path is specified - the cli configs run name is ignored, instead the run name
        # is determined by the saved config
        effective_run_name = download_run(run_path=wandb_run_path, replace=True)

    print('Run name', effective_run_name)
    if run_man.run_folder_exists(effective_run_name):
        print('Run folder already exists, reading its config')
        run_man.load_config(effective_run_name)

    return effective_run_name


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--config', type=str, help='Config file that will be used', required=True)
    parser.add_argument('-g', '--ngpus', type=int, help='Number of GPUs available', required=False)
    parser.add_argument('-s', '--stagger_number', type=int, required=False, default=-1,
                        help='Runs with this flag may only start when Time(S) % 10 == stagger_number')

    return parser.parse_args()


def stagger(stagger_number: int, stagger_time=6):
    """
    Used to make sure runs are not executed at exactly the same time

    @param stagger_number: the index of the current run in the stagger process
    @param stagger_time: amount of time there should be between runs (in seconds)
    @return:
    """
    if stagger_number == -1:
        print('no stagger number provided')
        return
    # the run has been given a stagger number
    # spin until the program is allowed to run
    print(f'using stagger number {stagger_number} and stagger time {stagger_time}')
    while int(time.time()) % (10 * stagger_time) != stagger_number * stagger_time:
        print(f'spinning t={int(time.time())}')
        time.sleep(0.5)
    print(f'staggered run until t={int(time.time())}')
