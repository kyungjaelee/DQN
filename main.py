import os
from gymhelpers import ExperimentsManager

env_name = "Acrobot-v1"
# env_name = "CartPole-v1"
# env_name = "MountainCar-v0"
gym_stats_dir_prefix = os.path.join('Gym_stats', env_name)
figures_dir = 'Figures'
api_key = '###'
alg_id = '###'


n_ep = 1000
n_exps = 1

expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=[256, 512],
                             figures_dir=figures_dir, discount=0.99, decay_eps=0.99, eps_min=1E-4, learning_rate=1E-3,
                             decay_lr=False, max_step=500, replay_memory_max_size=1000000, ep_verbose=False,
                             exp_verbose=True, batch_size=64, upload_last_exp=False, double_dqn=False,
                             target_params_update_period_steps=1000, replay_period_steps=4, min_avg_rwd=-78,
                             per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
                             per_beta0=0.1,
                             results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,strategy="softmax")
_, _, Rwd_per_ep_v, Loss_per_ep_v = expsman.run_experiments(n_exps=n_exps, n_ep=n_ep, stop_training_min_avg_rwd=-64, plot_results=True)
# expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=[256, 512],
#                              figures_dir=figures_dir, discount=0.99, decay_eps=0.99, eps_min=1E-4, learning_rate=3E-4,
#                              decay_lr=False, max_step=500, replay_memory_max_size=1000000, ep_verbose=False,
#                              exp_verbose=True, batch_size=64, upload_last_exp=True, double_dqn=False,
#                              target_params_update_period_steps=1000, replay_period_steps=4, min_avg_rwd=475,
#                              per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
#                              per_beta0=0.4,
#                              results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id,strategy="sparsemax")
# expsman.run_experiments(n_exps=n_exps, n_ep=n_ep, stop_training_min_avg_rwd=485, plot_results=False)
# expsman = ExperimentsManager(env_name=env_name, agent_value_function_hidden_layers_size=[256, 512],
#                              figures_dir=figures_dir, discount=0.99, decay_eps=0.995, eps_min=1E-4, learning_rate=1E-4,
#                              decay_lr=False, max_step=500, replay_memory_max_size=1000000, ep_verbose=False,
#                              exp_verbose=True, batch_size=64, upload_last_exp=True, double_dqn=False,
#                              target_params_update_period_steps=1000, replay_period_steps=4, min_avg_rwd=-110,
#                              per_proportional_prioritization=True, per_apply_importance_sampling=True, per_alpha=0.2,
#                              per_beta0=0.1,
#                              results_dir_prefix=gym_stats_dir_prefix, gym_api_key=api_key, gym_algorithm_id=alg_id)
# expsman.run_experiments(n_exps=n_exps, n_ep=n_ep, stop_training_min_avg_rwd=-100, plot_results=False)
