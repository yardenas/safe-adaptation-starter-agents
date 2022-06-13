#!/usr/bin/env python

import glob
import os
import tensorflow.compat.v1 as tf
from gym.wrappers import TimeLimit
import safe_adaptation_gym
from safe_adaptation_gym.benchmark import TASKS
from safe_rl.utils.logx import restore_tf_graph


def extract_by_name(string, name_list):
  for name in name_list:
    if name.lower() in string.lower():
      return name.lower()
  else:
    return ''


def load_policy(fpath, deterministic=False):
  params_dir = glob.glob(os.path.join(fpath, '**', 'simple_save*'))[0]
  # load the things!
  sess = tf.Session(graph=tf.Graph())
  model = restore_tf_graph(sess, params_dir)

  # get the correct op for executing actions
  if deterministic and 'mu' in model.keys():
    # 'deterministic' is only a valid option for SAC policies
    print('Using deterministic action op.')
    action_op = model['mu']
  else:
    print('Using default action op.')
    action_op = model['pi']

  # make function for producing an action given a single state
  get_action = lambda x: sess.run(
      action_op, feed_dict={model['x']: x[None, :]})[0]

  robot = extract_by_name(fpath, ['Point', 'Car', 'Doggo'])
  task_name = extract_by_name(fpath, TASKS.keys())
  env = safe_adaptation_gym.make(
      robot,
      task_name,
      seed=10,
      render_options=dict(camera_id='fixedfar', height=480, width=480))
  env = TimeLimit(env, 1000)
  return env, get_action, sess, (task_name, robot)
