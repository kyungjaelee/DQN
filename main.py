import os
from gymhelpers import ExperimentsManager
import scipy.io as scipyio
from collections import defaultdict

strategies = ["sparsemax","softmax","epsilon"]
backuprules = ["sparsebellman","softbellman","bellman"]

# env_name = "Acrobot-v1"
# gym_stats_dir_prefix = os.path.join('Gym_stats', env_name)
# figures_dir = 'Figures'
# api_key = '###'
# alg_id = '###'
#
# n_ep = 5000
# n_exps = 1
#
# data = {}
# for strategy in strategies:
#     print("Problem: {}, Strategy: {}".format(env_name,strategy))
#     expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=[256, 512],
#                                  figures_dir=figures_dir, discount=0.99, decay_eps=0.99, eps_min=1E-4, learning_rate=1E-3,
#                                  decay_lr=False, max_step=500, replay_memory_max_size=1000000, ep_verbose=False,
#                                  exp_verbose=True, batch_size=64, upload_last_exp=False, double_dqn=False,
#                                  target_params_update_period_steps=1000, replay_period_steps=4, min_avg_rwd=-78,
#                                  per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
#                                  per_beta0=0.1,
#                                  results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,strategy=strategy)
#     _, _, Rwd_per_ep_v, Loss_per_ep_v = expsman.run_experiments(n_exps=n_exps, n_ep=n_ep, stop_training_min_avg_rwd=-64, plot_results=False)
#     data[strategy] = {"reward_list":Rwd_per_ep_v,"loss_list":Loss_per_ep_v}
#
# scipyio.savemat(env_name+".mat", data)
# print("{} is finished".format(env_name))


# env_name = "InvertedPendulum-v1"; min_avg_rwd = 930; stop_training_min_avg_rwd = 950; action_res = 1001; n_ep = 2000; layers_size = [512, 512]
# env_name = "Pendulum-v0"; min_avg_rwd = -135; stop_training_min_avg_rwd = -125; action_res = 2001; layers_size = [512, 512]
env_name = "Reacher-v1"; min_avg_rwd = -3.5; stop_training_min_avg_rwd = -3.75; action_res = [51, 51]; n_ep = 10000; layers_size = [512, 512]
# env_name = "Swimmer-v1"; min_avg_rwd = 350; stop_training_min_avg_rwd = 360; action_res = [51, 51]; n_ep = 5000; layers_size = [512, 512]
# env_name = "Hopper-v1"; min_avg_rwd = 3700; stop_training_min_avg_rwd = 3800; action_res = [21, 21, 21]; n_ep = 5000; layers_size = [512, 512, 512]

gym_stats_dir_prefix = os.path.join('Gym_stats', env_name)
figures_dir = 'Figures'
api_key = '###'
alg_id = '###'

n_exps = 1
temperature = 1

data = defaultdict(lambda : defaultdict(lambda : None))
for backuprule in backuprules:
    for strategy in strategies:
        print("Problem: {}, Strategy: {}, Backup: {}".format(env_name,strategy,backuprule))
        # expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=layers_size,
        #                              figures_dir=figures_dir, discount=0.99, decay_eps=0.995, eps_min=1E-4, learning_rate=3E-3,
        #                              decay_lr=True, max_step=1000, replay_memory_max_size=100000, ep_verbose=False,
        #                              exp_verbose=True, learning_rate_end=3E-4, batch_size=64, upload_last_exp=False, double_dqn=True, dueling=False,
        #                              target_params_update_period_steps=320, replay_period_steps=4, min_avg_rwd=min_avg_rwd,
        #                              per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
        #                              per_beta0=0.4,
        #                              results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,
        #                              strategy=strategy,backuprule=backuprule,temperature=temperature,action_res=31)

        #########################
        ### Inverted Pendulum ###
        #########################

        # expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=layers_size,
        #                                      figures_dir=figures_dir, discount=0.99, decay_eps=0.995, eps_min=1E-4, learning_rate=3E-4,
        #                                      decay_lr=True, max_step=1000, replay_memory_max_size=100000, ep_verbose=False,
        #                                      exp_verbose=True, learning_rate_end=3E-5, batch_size=64, upload_last_exp=False, double_dqn=True, dueling=False,
        #                                      target_params_update_period_steps=50, replay_period_steps=4, min_avg_rwd=min_avg_rwd,
        #                                      per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
        #                                      per_beta0=0.4,
        #                                      results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,
        #                                      strategy=strategy,backuprule=backuprule,temperature=temperature,action_res=31)

        ################
        ### Pendulum ###
        ################

        # expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=layers_size,
        #                                       figures_dir=figures_dir, discount=0.99, decay_eps=0.995, eps_min=1E-4, learning_rate=3E-4,
        #                                       decay_lr=True, max_step=1000, replay_memory_max_size=100000, ep_verbose=False,
        #                                       exp_verbose=True, learning_rate_end=3E-5, batch_size=64, upload_last_exp=False, double_dqn=True, dueling=False,
        #                                       target_params_update_period_steps=100, replay_period_steps=4, min_avg_rwd=min_avg_rwd,
        #                                       per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
        #                                       per_beta0=0.4,
        #                                       results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,
        #                                       strategy=strategy,backuprule=backuprule,temperature=temperature,action_res=action_res)

        ########################
        ### Reacher Settings ###
        ########################

        expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=layers_size,
                                              figures_dir=figures_dir, discount=0.99, decay_eps=0.999, eps_min=1E-4, learning_rate=3E-4,
                                              decay_lr=True, max_step=10000, replay_memory_max_size=10000, ep_verbose=False,
                                              exp_verbose=True, learning_rate_end=3E-5, batch_size=64, upload_last_exp=False, double_dqn=True, dueling=False,
                                              target_params_update_period_steps=25, replay_period_steps=4, min_avg_rwd=min_avg_rwd,
                                              per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
                                              per_beta0=0.4,
                                              results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,
                                              strategy=strategy,backuprule=backuprule,temperature=temperature,action_res=action_res)

        ########################
        ### Swimmer Settings ###
        ########################

        # expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=layers_size,
        #                                       figures_dir=figures_dir, discount=0.99, decay_eps=0.999, eps_min=1E-4, learning_rate=3E-4,
        #                                       decay_lr=True, max_step=5000, replay_memory_max_size=10000, ep_verbose=False,
        #                                       exp_verbose=True, learning_rate_end=3E-5, batch_size=64, upload_last_exp=False, double_dqn=True, dueling=False,
        #                                       target_params_update_period_steps=25, replay_period_steps=4, min_avg_rwd=min_avg_rwd,
        #                                       per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
        #                                       per_beta0=0.4,
        #                                       results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,
        #                                       strategy=strategy,backuprule=backuprule,temperature=temperature,action_res=action_res)

        ########################
        ### Hopper Settings  ###
        ########################

        # expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=layers_size,
        #                                       figures_dir=figures_dir, discount=0.99, decay_eps=0.995, eps_min=1E-4, learning_rate=3E-4,
        #                                       decay_lr=True, max_step=2500, replay_memory_max_size=10000, ep_verbose=False,
        #                                       exp_verbose=True, learning_rate_end=3E-5, batch_size=64, upload_last_exp=False, double_dqn=True, dueling=False,
        #                                       target_params_update_period_steps=100, replay_period_steps=4, min_avg_rwd=min_avg_rwd,
        #                                       per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
        #                                       per_beta0=0.4,
        #                                       results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,
        #                                       strategy=strategy,backuprule=backuprule,temperature=temperature,action_res=action_res)

        _, _, Rwd_per_ep_v, Loss_per_ep_v = expsman.run_experiments(n_exps=n_exps, n_ep=n_ep, stop_training_min_avg_rwd=stop_training_min_avg_rwd, plot_results=False)
        data[strategy][backuprule] = {"reward_list":Rwd_per_ep_v,"loss_list":Loss_per_ep_v}

scipyio.savemat(env_name+'_'+str(temperature)+".mat", data)
print("{} is finished".format(env_name))

