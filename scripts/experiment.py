#!/usr/bin/env python
from safe_adaptation_gym import make

import safe_rl

from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork


def main(robot, task, algo, seed, exp_name, cpu):
  algo = algo.lower()
  # Hyperparameters
  if robot == 'Doggo':
    num_steps = 1e8
    steps_per_epoch = 60000
  else:
    num_steps = 1e7
    steps_per_epoch = 30000
  epochs = int(num_steps / steps_per_epoch)
  save_freq = 50
  target_kl = 0.01
  cost_lim = 25
  # Fork for parallelizing
  mpi_fork(cpu)
  # Prepare Logger
  # Algo and Env
  algo_fn = eval('safe_rl.' + algo)
  exp_name = exp_name or (algo.lower() + '_' + robot.lower() + '_' +
                          task.lower())
  logger_kwargs = setup_logger_kwargs(exp_name, seed)

  # Algo and Env
  algo = eval('safe_rl.' + algo)
  algo(
      env_fn=lambda: make(robot.lower(), task),
      ac_kwargs=dict(hidden_sizes=(256, 256),),
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      save_freq=save_freq,
      target_kl=target_kl,
      cost_lim=cost_lim,
      seed=seed,
      logger_kwargs=logger_kwargs)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--robot', type=str, default='Point')
  parser.add_argument('--task', type=str, default='GoToGoal')
  parser.add_argument('--algo', type=str, default='ppo')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--exp_name', type=str, default='')
  parser.add_argument('--cpu', type=int, default=1)
  args = parser.parse_args()
  exp_name = args.exp_name if not (args.exp_name == '') else None
  main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu)
