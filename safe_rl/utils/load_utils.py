#!/usr/bin/env python

import joblib
import os
import os.path as osp
import tensorflow.compat.v1 as tf
from safe_adaptation_gym import safe_adaptation_gym
from safe_adaptation_gym.benchmark import _TASKS
from safe_rl.utils.logx import restore_tf_graph


def extract_by_name(string, name_list):
  for name in name_list:
    if name.lower() in string.lower():
      return name.lower()
  else:
    return ''


def load_policy(fpath, itr='last', deterministic=False):

  # handle which epoch to load from
  if itr == 'last':
    saves = [
        int(x[11:])
        for x in os.listdir(fpath)
        if 'simple_save' in x and len(x) > 11
    ]
    itr = '%d' % max(saves) if len(saves) > 0 else ''
  else:
    itr = '%d' % itr

  # load the things!
  sess = tf.Session(graph=tf.Graph())
  model = restore_tf_graph(sess, osp.join(fpath, 'simple_save' + itr))

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
  task_name = extract_by_name(fpath, _TASKS.keys())
  env = safe_adaptation_gym.make(task_name, robot)

  return env, get_action, sess
