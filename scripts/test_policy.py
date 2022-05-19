#!/usr/bin/env python

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger


def make_video(frames, framerate=30):
  height, width, _ = frames[0].shape
  dpi = 70
  fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
  ax.set_axis_off()
  ax.set_aspect('equal')
  ax.set_position([0, 0, 1, 1])
  im = ax.imshow(frames[0])

  def update(frame):
    im.set_data(frame)
    return [im]

  interval = 1000 / framerate
  anim = animation.FuncAnimation(
      fig=fig,
      func=update,
      frames=frames,
      interval=interval,
      blit=True,
      repeat=False)
  return anim


def run_policy(env,
               get_action,
               max_ep_len=None,
               num_episodes=100,
               render=True,
               outfile='output.gif'):

  assert env is not None, \
      "Environment not found!\n\n It looks like the environment wasn't saved, " + \
      "and we can't run the agent in it. :("

  logger = EpochLogger()
  o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
  frames = []
  while n < num_episodes:
    if render:
      frames.append(
          env.unwrapped.render(camera_id='fixedfar', height=480, width=480))

    a = get_action(o)
    a = np.clip(a, env.action_space.low, env.action_space.high)
    o, r, d, info = env.step(a)
    ep_ret += r
    ep_cost += info.get('cost', 0)
    ep_len += 1

    if d or (ep_len == max_ep_len):
      logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
      print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d' %
            (n, ep_ret, ep_cost, ep_len))
      o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
      n += 1
      if render:
        video = make_video(frames)
        video.save(
            '{}_{}'.format(n, outfile), writer=animation.PillowWriter(fps=30))
      frames.clear()

  logger.log_tabular('EpRet', with_min_and_max=True)
  logger.log_tabular('EpCost', with_min_and_max=True)
  logger.log_tabular('EpLen', average_only=True)
  logger.dump_tabular()


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--fpath', type=str, nargs='*')
  parser.add_argument('--len', '-l', type=int, default=0)
  parser.add_argument('--episodes', '-n', type=int, default=100)
  parser.add_argument('--norender', '-nr', action='store_true')
  parser.add_argument('--deterministic', '-d', action='store_true')
  args = parser.parse_args()
  for dir_ in args.fpath:
    env, get_action, sess, (robot,
                            task_name) = load_policy(dir_, args.deterministic)
    run_policy(
        env,
        get_action,
        args.len,
        args.episodes,
        not (args.norender),
        outfile=task_name + '_' + robot + '.gif')
