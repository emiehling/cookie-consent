"""A simple recommender system agent that recommends random slates."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from absl import logging

import numpy as np

from recsim import agent


class RandomAgentMulti(agent.AbstractMultiUserEpisodicRecommenderAgent):
  """An agent that recommends a random slate of documents to multiple users."""

  def __init__(self, action_space, random_seed=0):
    super(RandomAgentMulti, self).__init__(action_space)
    self._rng = np.random.RandomState(random_seed)

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: Unused.
      observation: A dictionary that includes the most recent observation.
        Should include 'doc' field that includes observation of all candidates.

    Returns:
      slate: An integer array of size _slate_size, where each element is an
        index into the list of doc_obs
    """
    del reward  # Unused argument.
    doc_obs = observation['doc']

    # Simulate a random slate for each user
    doc_ids = list(range(len(doc_obs)))
    slates = []
    for i in range(self._num_users):
      self._rng.shuffle(doc_ids)
      slates.append(doc_ids[:self._slate_size])

    return slates
    # logging.debug('Recommended slate: %s', slate)
