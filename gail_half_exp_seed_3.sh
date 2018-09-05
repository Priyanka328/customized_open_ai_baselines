python -m baselines.gail.run_mujoco --gpu_id 1 --env_id "HalfCheetah-v2" --seed 3 --traj_limitation 1 --expert_path data/stochastic.trpo.HalfCheetah.0.00.npz
python -m baselines.gail.run_mujoco --gpu_id 1 --env_id "HalfCheetah-v2" --seed 3 --traj_limitation 5 --expert_path data/stochastic.trpo.HalfCheetah.0.00.npz
python -m baselines.gail.run_mujoco --gpu_id 1 --env_id "HalfCheetah-v2" --seed 3 --traj_limitation 10 --expert_path data/stochastic.trpo.HalfCheetah.0.00.npz
python -m baselines.gail.run_mujoco --gpu_id 1 --env_id "HalfCheetah-v2" --seed 3 --traj_limitation 25 --expert_path data/stochastic.trpo.HalfCheetah.0.00.npz
python -m baselines.gail.run_mujoco --gpu_id 1 --env_id "HalfCheetah-v2" --seed 3 --traj_limitation 50 --expert_path data/stochastic.trpo.HalfCheetah.0.00.npz

