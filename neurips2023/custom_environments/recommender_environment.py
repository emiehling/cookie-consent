from gym import spaces

from recsim.user import AbstractUserState, AbstractUserSampler, AbstractResponse, AbstractUserModel
from recsim.document import AbstractDocument, AbstractDocumentSampler

from recsim import choice_model
from recsim.simulator.environment import MultiUserEnvironment
from recsim.simulator import recsim_gym

from custom_utils import *

import time


class RSAdvertisement(AbstractDocument):

    def __init__(self,
                 ad_id,
                 # quality,
                 topic,
                 num_topics):
        # self.quality = quality
        self.topic = topic
        self.num_topics = num_topics
        AbstractDocument.__init__(self,
                                  doc_id=ad_id)

    def create_observation(self):
        # return {'quality': np.array(self.quality), 'topic': self.topic}
        return {'topic': self.topic}

    def observation_space(self):
        return spaces.Dict({
            # 'quality':  # potentially add additional features
            #     spaces.Box(
            #         low=0.0, high=np.inf, shape=tuple(), dtype=np.float32),
            'topic':
                spaces.Discrete(self.num_topics)
        })


class RSAdvertisementSampler(AbstractDocumentSampler):
    """
    Generates an ad instance given ad features distributions.

    """

    def __init__(self,
                 ad_sampling_params,
                 doc_ctor=RSAdvertisement,
                 **kwargs):
        self._num_topics = ad_sampling_params['num_topics']
        self._topic_dist = ad_sampling_params['topic_distribution']
        # self._topic_quality_mean = topic_quality_mean
        # self._topic_quality_stddev = topic_quality_stddev
        AbstractDocumentSampler.__init__(self,
                                         doc_ctor,
                                         **kwargs)  # doc_ctor required by AbstractDocumentSampler
        # super(AdvertisementSampler, self).__init__(doc_ctor, **kwargs)
        self._ad_count = 0      # keeps track of ad index

    # @property
    # def num_clusters(self):
    #     return self._number_of_topics

    def sample_document(self):
        # Samples the ad topic from the topic distribution.

        ad_features = {}
        ad_features['ad_id'] = self._ad_count
        self._ad_count += 1  # increment ad counter for next call

        if self._topic_dist['type'] == 'uniform':
            topic = self._rng.randint(self._num_topics)
        else:
            raise NotImplementedError(f"Specified topic distribution type '{self._topic_dist['type']}' is not implemented.")
        ad_features['topic'] = topic

        ad_features['num_topics'] = self._num_topics

        # ad_quality = (
        #     self._rng.lognormal(
        #         mean=self._topic_quality_mean[topic],
        #         sigma=self._topic_quality_stddev[topic]))
        # ad_features['quality'] = ad_quality

        return self._doc_ctor(**ad_features)


class RSUserState(AbstractUserState):
    '''
    Class to represent user attributes.

    '''

    def __init__(self,
                 user_cookie,
                 num_cookies,
                 user_cohort,
                 user_consent,
                 topic_affinity,
                 user_choice_model,
                 time_budget):
        super().__init__()

        # User parameters
        self.cookie = user_cookie
        self.num_cookies = num_cookies
        self.cohort = user_cohort
        self.consent = user_consent
        self.topic_affinity = topic_affinity,
        self.choice_features = {feature: value for feature, value in user_choice_model['choice_features'].items()}
        self.time_budget = time_budget  # the only (dynamic) state variable

    def create_observation(self):
        # This is the information about the user's features that is revealed to the agent before the interaction.
        # The agent is informed of the user's consent decision and their cookie value if consent is provided (cookie is
        # otherwise set to -1)
        observation = {
            'cookie': self.cookie if self.consent == 1 else -1,
            'consent': self.consent
        }
        return observation

    def score_document(self, ad_obs):
        return self.topic_affinity[0][ad_obs['topic']]  # simply returns the affinity of the topic

    def observation_space(self):
        # The cookie space includes -1 for no consent, and 0 to num_cookies-1 for consent
        # Note: Box was used (even though cookies are integers) since Discrete doesn't allow for negative numbers
        return spaces.Dict({
            'consent': spaces.Discrete(2),  # Binary consent variable
            'cookie': spaces.Box(low=np.array([-1]), high=np.array([self.num_cookies-1]), dtype=np.int32)
        })


class UserStateSampler(AbstractUserSampler):
    """
    Generates a user instance given user features distributions.

    """

    def __init__(self,
                 user_ctor,
                 user_cookie_cohort_distribution,
                 user_consent_distribution,
                 topic_affinity_means,
                 topic_affinity_stds,
                 user_choice_model,
                 time_budget,
                 **kwargs):
        self._state_parameters = {'user_cookie_cohort_dist': user_cookie_cohort_distribution,
                                  'user_consent_dist': user_consent_distribution,
                                  'num_cookies': len(user_cookie_cohort_distribution),
                                  'num_cohorts': len(user_cookie_cohort_distribution[0]),
                                  'topic_affinity_means': topic_affinity_means,
                                  'topic_affinity_stds': topic_affinity_stds,
                                  'user_choice_model': user_choice_model,
                                  'time_budget': time_budget,
                                  }
        # AbstractUserSampler.__init__(self,
        #                              user_ctor=user_ctor,
        #                              **kwargs)
        super().__init__(user_ctor, **kwargs)
        # super(UserStateSampler, self).__init__(user_ctor, **kwargs)

    def generate_user_cookie_and_cohort(self, user_cookie_cohort_dist):
        # Sample cookie first from p(c), then cohort from p(d|c) (since p(c,d) = p(c)p(d|c))
        user_cookie_cohort_dist = np.array(user_cookie_cohort_dist)
        marginal_dist_on_cookies = np.sum(user_cookie_cohort_dist, axis=1)  # form p(c)
        cookie = self._rng.choice(self._state_parameters['num_cookies'],
                                  p=marginal_dist_on_cookies)
        conditional_dist_on_cohorts_given_cookie = \
            user_cookie_cohort_dist[cookie,:] / sum(user_cookie_cohort_dist[cookie,:])
        user_cohort = self._rng.choice(self._state_parameters['num_cohorts'],
                                       p=conditional_dist_on_cohorts_given_cookie)  # sample p(d|c)

        return cookie, user_cohort

    def generate_user_consent(self, user_cohort):
        consent = self._rng.binomial(n=1, p=self._state_parameters['user_consent_dist'][user_cohort])
        return consent

    def generate_user_topic_affinity(self, user_cohort):
        user_topic_affinity = self._rng.lognormal(
            mean=self._state_parameters['topic_affinity_means'][user_cohort, :],
            sigma=self._state_parameters['topic_affinity_stds'][user_cohort, :])
        return user_topic_affinity

    def sample_user(self):
        # Note: These are the state parameters that are sampled. The state parameters that appear in _state_parameters
        # in init are fixed.

        # Sample user cookie and cohort
        cookie, cohort = self.generate_user_cookie_and_cohort(self._state_parameters['user_cookie_cohort_dist'])
        consent = self.generate_user_consent(cohort)
        topic_affinity = self.generate_user_topic_affinity(cohort)

        return self._user_ctor(
            user_cookie=cookie,
            num_cookies=self._state_parameters['num_cookies'],
            user_cohort=cohort,
            user_consent=consent,
            topic_affinity=topic_affinity,
            # topic_affinity_means=self._state_parameters['topic_affinity_means'],
            # topic_affinity_stddev=self._state_parameters['topic_affinity_stds'],
            user_choice_model=self._state_parameters['user_choice_model'],
            time_budget=self._state_parameters['time_budget']
        )


class RSUserResponse(AbstractResponse):
    ''' Class to create a user response object.

    '''

    def __init__(self,
                 clicked: bool = False,
                 topic: int = 0,
                 num_topics: int = 0):
        """Creates a new user response object for an advertisement.

        Args:
          clicked: A boolean indicating whether the advertisement was clicked.
          topic: An integer for the topic of the advertisement.
        """
        super().__init__()
        self.clicked = clicked
        self.topic = topic
        self.num_topics = num_topics

    def create_observation(self):
        return {
            'clicked': int(self.clicked),
            'topic': self.topic
        }

    @classmethod
    def set_num_topics(cls, num_topics):
        cls.num_topics = num_topics

    @classmethod
    def response_space(cls):
        if cls.num_topics is None:
            raise ValueError("num_topics is not set for RSUserResponse")
        return spaces.Dict({
            'clicked': spaces.Discrete(2),
            'topic': spaces.Discrete(cls.num_topics)
        })


class RSUserModel(AbstractUserModel):
    """ Class to create a user model

    The user's choice is characterized by a vector of affinity scores of dimension NUM_TOPICS. When presented with an
    ad, the user's score is computed by simply extracting the affinity corresponding to the topic of the ad.

    """

    def __init__(self,
                 slate_size,
                 user_state_ctor=RSUserState,
                 response_model_ctor=RSUserResponse,
                 ad_sampling_params=None,
                 user_params=None,
                 seed=0):
        """ Initializes a new user model.

        Args
          slate_size: integer representing the size of the slate
          choice_model_ctor: A constructor to create a new user choice object.
          response_model_ctor: A constructor to create a new user response object
          user_state_ctor: A constructor to create a new user state object.
          no_click_mass: A float that influences the probability of a no-click.
          seed: An integer used as the seed of the choice model.
        """
        if 'user_choice_model' in user_params:
            choice_model_ctor = getattr(choice_model, user_params['user_choice_model']['type'])
            self.choice_model = choice_model_ctor(user_params['user_choice_model']['choice_features'])
        else:
            raise Exception("Must specify a choice model.")

        user_sampler = UserStateSampler(
            user_ctor=user_state_ctor,
            seed=seed,
            **user_params
        )
        super().__init__(
            response_model_ctor=response_model_ctor,
            user_sampler=user_sampler,
            slate_size=slate_size,
        )

        # AbstractUserModel.__init__(self,
        #                            response_model_ctor=RSUserResponse,
        #                            user_sampler=user_state_ctor,
        #                            slate_size=slate_size

        self._user_state_ctor = user_state_ctor
        self.num_topics = ad_sampling_params['num_topics']

    def simulate_response(self, slate_ads):
        '''
        This method implements the user's choice model.

        :param slate_topics: the set of ad topics shown to the user
        :return: populates responses object based on which ad the user selected
        '''

        # List of empty responses
        responses = [
            self._response_model_ctor(
                clicked=False,
                topic=ad.topic,
                num_topics=self.num_topics
            ) for ad in slate_ads
        ]

        self.choice_model.score_documents(
            self._user_state, [ad.create_observation() for ad in slate_ads])
        selected_index = self.choice_model.choose_item()

        # Record feature of ad that the user was queried with
        for i, response in enumerate(responses):
            response.topic = slate_ads[i].topic

        # Return the default (no-click) responses if no ad was clicked
        if selected_index is None:
            return responses

        # If some ad was picked (selected_index), then update response object
        self.generate_response(slate_ads[selected_index],
                               responses[selected_index])
        return responses

    def generate_response(self, ad, response):
        response.clicked = True
        response.topic = ad.topic

    def update_state(self, slate_documents, responses):
        # Simple state update (decrement time budget). Can be augmented to be more complex.
        self._user_state.time_budget -= 1

    def is_terminal(self):
        # End session if user's time budget goes to zero
        return self._user_state.time_budget <= 0


def agent_reward_function(responses):
    # Definition of the recommender agent's reward function
    reward = 0.0
    for response in responses:
        reward += response[0].clicked  # click reward
        # if response.clicked:  # engagement reward (not modeled)
        #     reward += response.engagement
    return reward


def validate_environment_config(env_config):
    required_keys = ['num_users', 'slate_size', 'users', 'ads', 'num_candidate_ads', 'resample_documents', 'seed']
    for key in required_keys:
        if key not in env_config:
            raise ValueError(f"Missing required environment config key: {key}")


def create_environment(env_config):
    """Creates a recommender with consent environment.

    - parameters of the environment are specified in experiment_params.get_experiments()
    """

    validate_environment_config(env_config)

    # Parameters of the ads and users
    ad_sampling_params = {
        'num_topics': env_config['ads']['num_topics'],
        'topic_distribution': env_config['ads']['topic_distribution']
    }
    user_params = {
        'user_cookie_cohort_distribution': env_config['users']['prior'],
        'user_consent_distribution': env_config['users']['consent_rates'],
        'topic_affinity_means': env_config['users']['topic_affinity_means'],
        'topic_affinity_stds': env_config['users']['topic_affinity_stds'],
        'time_budget': env_config['users']['time_budget'],
        'user_choice_model': env_config['users']['choice_model'],
    }
    RSUserResponse.set_num_topics(ad_sampling_params['num_topics'])  # to pass num_topics to class level (compatibility)

    # Define user models
    user_models = []
    for _ in range(env_config['num_users']):
        user_models.append(
            RSUserModel(
                slate_size=env_config['slate_size'],
                user_state_ctor=RSUserState,
                response_model_ctor=RSUserResponse,
                ad_sampling_params=ad_sampling_params,
                user_params=user_params,
                seed=env_config['seed'])
        )

    # Define advertisement sampler
    advertisement_sampler = RSAdvertisementSampler(
        ad_sampling_params=ad_sampling_params,
        seed=env_config.get('seed'))

    # Construct raw (multi-user) environment
    rs_env = MultiUserEnvironment(user_models,
                                  document_sampler=advertisement_sampler,
                                  num_candidates=env_config['num_candidate_ads'],
                                  slate_size=env_config['slate_size'],
                                  resample_documents=env_config['resample_documents'])

    return recsim_gym.RecSimGymEnv(raw_environment=rs_env,
                                   reward_aggregator=agent_reward_function)
                                   #total_clicks_reward,
                                   #custom_utils.aggregate_consent_metrics,
                                   #custom_utils.write_consent_metrics)
