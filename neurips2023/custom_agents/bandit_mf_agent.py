import numpy as np
from sklearn.decomposition import NMF  # Non-negative Matrix Factorization
from recsim.agent import AbstractEpisodicRecommenderAgent


class BanditMFAgent(AbstractEpisodicRecommenderAgent):
    def __init__(self,
                 agent_params,
                 observation_space,
                 action_space,
                 random_seed=0):
        super(BanditMFAgent, self).__init__(action_space)
        self._num_features = agent_params['num_features']
        self._epsilon = agent_params['epsilon']
        self._batch_size = agent_params['batch_size']
        self._rng = np.random.RandomState(random_seed)
        self._user_factors = None
        self._item_factors = None
        self._counter = 0
        self._training_round = True

    def _matrix_factorization(self, data):
        """Runs Matrix Factorization on the data."""
        model = NMF(n_components=self._n_features, random_state=self._rng)
        self._user_factors = model.fit_transform(data)
        self._item_factors = model.components_

    def set_training_round(self, is_training_round):
        """Sets the training round flag."""
        self._training_round = is_training_round

    def step(self, reward, observation):
        self._counter += 1
        if self._training_round or (self._counter % self._batch_size == 0):
            self._matrix_factorization(observation['data'])
            # todo:
            #  - assuming 'data' contains the user-item matrix
            #  - should actually include another function that extracts the response matrix (agent maintains this information)

        if self._rng.random() < self._epsilon:
            # Exploration: recommend a random slate of documents
            doc_ids = list(range(len(observation['doc'])))
            self._rng.shuffle(doc_ids)
            slate = doc_ids[:self._slate_size]
        else:
            # Exploitation: recommend documents based on learned factors
            user_vector = self._user_factors[observation['user_id']]  # assuming 'user_id' is a field in observation
            scores = np.dot(user_vector, self._item_factors)
            slate = np.argsort(scores)[-self._slate_size:]  # get indices of top scoring documents

        return slate

    def __str__(self):
        return 'BanditMFAgent'

# call via: agent = EpsGreedy(num_arms=10, params={"epsilon": 0.05}, seed=42)
# # Usage:
# agent = BanditMFAgent(action_space, n_features)
# agent.set_training_round(True)  # Set to training round
# # ... run training episodes ...
#
# agent.set_training_round(False)  # Set to non-training round
# # ... run non-training episodes ...