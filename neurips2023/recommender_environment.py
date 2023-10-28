from gym import spaces

from recsim.user import AbstractUserState, AbstractUserSampler, AbstractResponse, AbstractUserModel
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.document import AbstractDocument, AbstractDocumentSampler

from base_functions import *
import time

class Advertisement(AbstractDocument):

    def __init__(self,
                 ad_id,
                 # quality,
                 topic):
        # self.quality = quality
        self.topic = topic
        AbstractDocument.__init__(self,
                                  doc_id=ad_id)

    def create_observation(self):
        # return {'quality': np.array(self.quality), 'topic': self.topic}
        return {'topic': self.topic}

    @classmethod
    def observation_space(cls):
        return spaces.Dict({
            # 'quality':
            #     spaces.Box(
            #         low=0.0, high=np.inf, shape=tuple(), dtype=np.float32),
            'topic':
                spaces.Discrete(NUM_TOPICS)
        })



class AdvertisementSampler(AbstractDocumentSampler):
    """
    Generates an ad instance given ad features distributions.

    """

    def __init__(self,
               topic_distribution=(.3, .3, .4),
               # topic_quality_mean=(1., 1., 1.),
               # topic_quality_stddev=(.5, .5, .5),
               doc_ctor=Advertisement,
               **kwargs):
        self._number_of_topics = len(topic_distribution)
        self._topic_dist = topic_distribution
        # self._topic_quality_mean = topic_quality_mean
        # self._topic_quality_stddev = topic_quality_stddev
        AbstractDocumentSampler.__init__(self,
                                         doc_ctor,
                                         **kwargs)  # doc_ctor required by AbstractDocumentSampler
        # super(AdvertisementSampler, self).__init__(doc_ctor, **kwargs)
        self._ad_count = 0      # keeps track of ad index

    @property
    def num_clusters(self):
        return self._number_of_topics

    def sample_document(self):
        """Samples the topic and then samples document features given the topic."""
        ad_features = {}
        ad_features['ad_id'] = self._ad_count
        self._ad_count += 1
        topic = self._rng.choice(self._number_of_topics, p=self._topic_dist)
        # ad_quality = (
        #     self._rng.lognormal(
        #         mean=self._topic_quality_mean[topic],
        #         sigma=self._topic_quality_stddev[topic]))
        ad_features['topic'] = topic
        # ad_features['quality'] = ad_quality
        return self._doc_ctor(**ad_features)



class RSUserState(AbstractUserState):
    '''
    Contains the user state.

    '''

    def __init__(self, user_cookie, user_cohort, user_cookie_cohort_dist, user_consent, user_consent_dist,
                 user_topic_affinity, user_ad_means, user_ad_stddev, no_click_mass, time_budget):
        AbstractUserState.__init__(self)

        # User parameters
        self.user_cookie = user_cookie
        self.user_cohort = user_cohort
        self.user_cookie_cohort_dist = user_cookie_cohort_dist
        self.user_consent = user_consent
        self.user_consent_dist = user_consent_dist
        self.user_topic_affinity = user_topic_affinity
        self.user_ad_means = user_ad_means
        self.user_ad_stddev = user_ad_stddev
        self.no_click_mass = no_click_mass

        ## State variables
        self.time_budget = time_budget

    def create_observation(self):
        # This is the information about the user's features that is revealed to the agent over the interaction.
        # By assumption this is empty; only static attribute and behavioral information (clicks/engagement) is revealed
        return np.array([])

    @staticmethod
    def observation_space():
        # Empty observation space (since agent doesn't see anything about the user)
        return spaces.Box(shape=(0,), dtype=np.float32, low=0.0, high=np.inf)



class UserStateSampler(AbstractUserSampler):
    """
    Generates a user instance given user features distributions.

    """
    _state_parameters = None

    def __init__(self,
                 user_ctor=RSUserState,
                 no_click_mass=20.0,
                 time_budget=1000000,    # set to a high number for persistent users
                 user_cookie_cohort_distribution=((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
                 user_consent_distribution=(0.5, 0.5),
                 user_document_mean_affinity_matrix=((1., 1., 1.), (1., 1., 1.)),
                 user_document_stddev_affinity_matrix=((.5, .5, .5), (.5, .5, .5)),
                 **kwargs):
        self._state_parameters = {'no_click_mass': no_click_mass,
                                  'time_budget': time_budget,
                                  'user_cookie_cohort_dist': user_cookie_cohort_distribution,
                                  'user_consent_dist': user_consent_distribution,
                                  'user_ad_means': user_document_mean_affinity_matrix,
                                  'user_ad_stddev': user_document_stddev_affinity_matrix
                                  }
        self._number_of_user_cookies = len(user_cookie_cohort_distribution)
        self._number_of_user_types = len(user_cookie_cohort_distribution[0])
        AbstractUserSampler.__init__(self,
                                     user_ctor=user_ctor,
                                     **kwargs)
        # super(UserStateSampler, self).__init__(user_ctor, **kwargs)

    def generate_user_cookie_and_cohort(self, user_cookie_cohort_dist):
        # Sample cookie first from p(c), then cohort from p(d|c) (since p(c,d) = p(c)p(d|c))
        user_cookie_cohort_dist = np.array(user_cookie_cohort_dist)
        marginal_dist_on_cookies = np.sum(user_cookie_cohort_dist, axis=1)       # form marginal distribution on cookies, p(c)
        cookie = self._rng.choice(self._number_of_user_cookies, p=marginal_dist_on_cookies)
        conditional_dist_on_cohorts_given_cookie = user_cookie_cohort_dist[cookie,:] / sum(user_cookie_cohort_dist[cookie,:])
        user_cohort = self._rng.choice(self._number_of_user_types, p=conditional_dist_on_cohorts_given_cookie) # sample from conditional distribution p(d|c)

        return cookie, user_cohort

    def generate_user_consent(self, user_cohort):
        consent = self._rng.binomial(n=1, p=self._state_parameters['user_consent_dist'][user_cohort])
        return consent

    def generate_user_topic_affinity(self, user_cohort):
        user_topic_affinity = self._rng.lognormal(
            mean=self._state_parameters['user_ad_means'][:, user_cohort],
            sigma=self._state_parameters['user_ad_stddev'][:, user_cohort])
        return user_topic_affinity

    def sample_user(self):
        # Note: These are the state parameters that are sampled. The state parameters that appear in _state_parameters
        # in init are fixed.

        # Sample user cookie and cohort
        cookie, cohort = self.generate_user_cookie_and_cohort(self._state_parameters['user_cookie_cohort_dist'])
        self._state_parameters['user_cookie'] = cookie
        self._state_parameters['user_cohort'] = cohort

        # Sample user consent decision
        self._state_parameters['user_consent'] = self.generate_user_consent(self._state_parameters['user_cohort'])

        # Sample user advertisement affinity given user cohort
        self._state_parameters['user_topic_affinity'] = self.generate_user_topic_affinity(self._state_parameters['user_cohort'])

        return self._user_ctor(**self._state_parameters)



class RSUserResponse(AbstractResponse):
    '''
    Creates a user response object.

    '''

    def __init__(self,
                 click=False,
                 topic=0):
        self.click = click
        self.topic = topic

    def create_observation(self):

        return {
            'click': int(self.click),
            'topic': self.topic
        }

    @classmethod
    def response_space(cls):

        return spaces.Dict({
            'click':
                spaces.Discrete(2),
            'topic':
                spaces.Discrete(NUM_TOPICS)
        })


class RSUserModel(AbstractUserModel):
    """
    User model

    The user's choice is characterized by a vector of affinity scores of dimension NUM_TOPICS. When presented with an
    ad, the user computes its score simply by extracting the affinity corresponding to the topic of the ad.

    """

    def __init__(self,
                 slate_size,
                 user_state_ctor,
                 choice_model_ctor=MultinomialLogitChoiceModel,
                 response_model_ctor=None,
                 seed=int(time.time())):

        AbstractUserModel.__init__(self,
                                   response_model_ctor=RSUserResponse,
                                   user_sampler=user_state_ctor,
                                   slate_size=slate_size)

        self._user_state_ctor = user_state_ctor
        self.choice_model = choice_model_ctor({
            'no_click_mass': self.user_state.no_click_mass,
        })

    def simulate_response(self, slate_ads):
        '''
        This method implements the user's choice model.

        :param slate_topics: the set of ad topics shown to the user
        :return: populates responses object based on which ad the user selected
        '''

        # List of empty responses
        responses = [self._response_model_ctor() for _ in slate_ads]

        # Compute logits and select document
        logits = np.array([])
        for ad in slate_ads:
            score = score_ad(self.user_state.user_topic_affinity[ad.topic])
            logits = np.append(logits, score)
        all_logit_scores = np.append(logits, self.user_state.no_click_mass)
        all_probs = all_logit_scores / np.sum(all_logit_scores)
        selected_index = np.random.choice(len(all_probs), p=all_probs)

        # Populate clicked item.
        if selected_index == len(all_probs) - 1:    # if no_click was chosen

            # Populate all responses with zero engagement
            for i, response in enumerate(responses):
                response.click = False
                response.topic = slate_ads[i].topic

        else:   # user clicked an item

            ad = slate_ads[selected_index]

            # Store user click as true
            responses[selected_index].click = True

            # Store topic of selected ad
            responses[selected_index].topic = ad.topic

        return responses

    def update_state(self, slate_documents, responses):
        # for ad, response in zip(slate_documents, responses):
        self.user_state.time_budget -= 1
        # return

    def is_terminal(self):
        # End session if user's time budget goes to zero
        return self.user_state.time_budget <= 0
