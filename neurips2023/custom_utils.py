import json
import numpy as np
# import matplotlib.pyplot as plt
from scipy.special import binom
# import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_topic_affinity_params(num_cohorts, num_topics, similarity):
    """
    Generates parameters for log-normal distributions representing topic affinities for different cohorts.

    This function creates a set of mean and standard deviation parameters for each cohort's topic affinity distribution.
    The means are generated such that the first cohort gets a baseline mean, and subsequent cohorts have their means
    adjusted based on the similarity coefficient. A higher similarity coefficient results in closer means between
    cohorts, while a lower coefficient increases the difference between their means. The standard deviations are assumed
    to be the same across all cohorts and topics.

    This function is a substitute for real topic affinity data across cohorts.

    Parameters:
    num_cohorts (int): number of cohorts to generate parameters for.
    num_topics (int): number of topics for which to generate affinity parameters.
    similarity_coefficient (float): value in  [0, 1] that controls the degree of similarity in means across cohorts.

    Returns:
    list: A list of dictionaries, where each dictionary contains 'means' and 'std_devs' keys with corresponding
          parameters for the log-normal distributions of topic affinities for a cohort.
    """
    baseline_means = np.random.rand(num_topics)
    delta = (np.random.rand(num_topics) - 0.5) / 5  # Small increment
    affinity_means = []
    affinity_stds = []
    for i in range(num_cohorts):
        if i == 0:
            means = baseline_means
        else:
            means = baseline_means + i * delta * (1 - similarity)
        std_devs = np.random.rand(num_topics)
        affinity_means.append(means)
        affinity_stds.append(std_devs)

    return np.array(affinity_means), np.array(affinity_stds)


# def get_affinity_means():
#     # Generates synthetic mean affinity values for each user attribute. This is a substitute for a dataset of affinity
#     # values for different demographics.
#     #
#     # Note: mean of normal distribution is mean_array[i] * num_topics
#
#     # Number of user attributes
#     num_attributes = len(TOPIC_MEANS)
#
#     weights = np.zeros((NUM_TOPICS, num_attributes))
#     for att in range(num_attributes):
#         s = np.digitize(np.random.normal(TOPIC_MEANS[att] * NUM_TOPICS, NUM_TOPICS / 10.0, 10000), range(NUM_TOPICS-1))
#         unique_values = np.unique(s)
#         counts = np.array([np.count_nonzero(s == val) for val in unique_values])
#         w = counts / max(counts)
#         for i, val in enumerate(unique_values):
#             weights[val, att] = w[i]
#
#     return weights


def kl_divergence(p, q):
    '''
    Compute KL divergence between p and q.

    :param p: list, possibly unnormalized.
    :param q: list, possibly unnormalized.
    :return: KL divergence between p and q.
    '''
    p = p/sum(p)
    q = q/sum(q)
    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))


def sample_user_responses(environment, action_space, num_samples):
    '''
    Samples the user pool for responses on randomly selected ads. Responses are used to run ridge regression on user and
    ad factor matrices.

    :param num_responses: number of desired responses from each user
    :param ad_pool: current set of candidate ads
    :return response_dict: set of responses (in a dictionary)
    '''
    response_dict = {}
    for j in range(num_samples):
        slates = action_space.sample()  # recommendations sampled uniformly at random
        slates_list = [slate[0] for slate in slates]
        observations, rewards, _, _ = environment.step(slates)
        for i in range(NUM_USERS):
            observations['response'][i][0]['ad_id'] = slates[i][0]
        response_dict[j] = [obs[0] for obs in observations['response']]

    return response_dict


class MF():

    def __init__(self, R, K, weights, non_nan_mask, learning_rate, reg, iterations):
        """
        Matrix factorization class.

        :param R: user-item rating matrix
        :param K: dimension of latent factors
        :param weights: confidence weights
        :param non_nan_mask: mask of entries that are not missing
        :param learning_rate: learning rate (for sgd)
        :param reg: regularization parameter

        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.weights = weights
        self.non_nan_mask = non_nan_mask
        self.learning_rate = learning_rate
        self.reg = reg
        self.iterations = iterations

    def train(self):
        # Initialize user and ad latent feature matrices
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if not np.isnan(self.R[i, j])
        ]

        # Train
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if mse < 0.001:
                break

        return training_process

    def mse(self):
        predicted = self.full_matrix()
        error = np.mean((self.weights[self.non_nan_mask] * (self.R[self.non_nan_mask] - predicted[self.non_nan_mask])) ** 2)
        return error

    def max_error(self):
        predicted = self.full_matrix()
        return np.max(self.weights[self.non_nan_mask] * np.abs(self.R[self.non_nan_mask] - predicted[self.non_nan_mask]))

    def sgd(self):
        """
        Perform stochastic gradient descent
        """

        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update user and item latent feature matrices
            self.P[i, :] += self.learning_rate * (e * self.Q[j, :] - self.reg * self.P[i,:])
            self.Q[j, :] += self.learning_rate * (e * self.P[i, :] - self.reg * self.Q[j,:])

    def get_rating(self, i, j):
        prediction = self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        return self.P.dot(self.Q.T)


def get_estimated_factors(responses, expected_probabilities):#, current_user_features, current_ad_features):
    '''
    Estimates latent factors from user responses via confidence-weighted MF procedure.

    :param responses: history of user-topic responses
    :param expected_probabilities: matrix represented current expected click probabilities, NUM_USERS x NUM_TOPICS
    :param n: current round
    :return: estimated user factor matrix, ad factor matrix
    '''

    # Extract matrix of user-ad response counts from responses argument
    user_topic_impressions_matrix = np.zeros((NUM_USERS, NUM_TOPICS))
    user_topic_response_matrix = np.zeros((NUM_USERS, NUM_TOPICS))
    user_topic_response_matrix[:] = np.NAN
    for i in range(NUM_USERS):

        # Extract responses for current user
        if type(responses) is dict:
            response_set_for_current_user = [resp[i] for resp in list(responses.values())]
        else:
            if len(responses) == 0: # If we have no responses, return random latent factor matrices
                return np.random.randn(NUM_USERS, NUM_FEATURES), np.random.randn(NUM_TOPICS, NUM_FEATURES)
            if type(responses[0]) is list:  # If responses are a list of lists (we have multiple rounds of responses)
                response_set_for_current_user = [resp[i] for resp in responses]
            else:   # If responses are a list of dictionaries (we have a single round of responses)
                response_set_for_current_user = responses[i]

        # Extract topic impressions and clicks for each response and use to populate user_topic_response_matrix
        for resp in response_set_for_current_user:
            topic_id = resp['topic']
            if np.isnan(user_topic_response_matrix[i, topic_id]):
                user_topic_response_matrix[i, topic_id] = resp['click']
            else:
                user_topic_response_matrix[i, topic_id] += resp['click']
            user_topic_impressions_matrix[i, topic_id] += 1

    # Check for missing values in user_ad_response_matrix
    if np.isnan(user_topic_response_matrix).any():
        print(' Response matrix contains missing elements. ')
        print(' Number of nan: ' + str(np.count_nonzero(np.isnan(user_topic_response_matrix))))

    # Create mask to filter out missing values
    filter_matrix = user_topic_response_matrix.copy()
    filter_matrix[np.isnan(filter_matrix)] = -1
    non_nan_mask = (filter_matrix != -1)

    # Compute confidence weights for matrix factorization procedure
    # confidences = (user_topic_response_matrix>=0).astype(int)
    confidences = binom(user_topic_impressions_matrix, user_topic_response_matrix) * \
                  expected_probabilities ** user_topic_response_matrix * \
                  (1 - expected_probabilities) ** (user_topic_impressions_matrix - user_topic_response_matrix)

    num_training_iterations = 5000#500000
    mf = MF(user_topic_response_matrix, NUM_FEATURES, confidences, non_nan_mask, learning_rate=0.01, reg=0.01, iterations=num_training_iterations)
    training_errors = mf.train()

    return mf.P, mf.Q, training_errors


def get_slates(user_factor_matrix, ad_factor_matrix, ad_pool):
    '''
    Computes ad recommendations (ad indices in current ad_pool) for each user using an epsilon-greedy bandit:
        - exploitation: ad index with maximal inner product of current factor matrix estimates wp 1-eps
        - exploration: chosen uniformly at random wp eps

    :param user_factor_matrix: current estimate of latent user factor matrix
    :param ad_factor_matrix: current estimate of latent ad factor matrix
    :param ad_pool: current pool of ads (list of topics)
    :return: recommendations across users
    '''

    exploration_probability = 0.1

    # Filter current ad factor matrix by current ad pool
    ad_factor_matrix_filtered = ad_factor_matrix[ad_pool, :]

    ad_recommendations = np.zeros(NUM_USERS, dtype=int)
    for i in range(NUM_USERS):
        scores = np.zeros(NUM_CANDIDATES)
        for a in range(NUM_CANDIDATES):
            scores[a] = np.dot(user_factor_matrix[i, :], ad_factor_matrix_filtered[a, :])
        explore = np.random.binomial(n=1, p=exploration_probability)
        if explore:
            ad_recommendations[i] = np.random.randint(NUM_CANDIDATES)
        else:   # exploit
            ad_recommendations[i] = np.argmax(scores)
    return [[rec] for rec in ad_recommendations]


def get_cohort_prior_beliefs(env):
    '''
    Cohort prior is computed using knowledge of consent signal and cookie value.

    :param env: given environment
    '''

    cohort_beliefs = np.zeros((NUM_USERS, NUM_COHORTS))

    for i in range(NUM_USERS):

        user_cookie_cohort_dist = env.environment.user_model[i].user_state.user_cookie_cohort_dist  # uniform across users
        user_marginal_on_cohort = np.sum(user_cookie_cohort_dist, axis=0)
        user_consent = env.environment.user_model[i].user_state.user_consent

        # Compute cohort distribution for user i
        if user_consent:  # then cookie is revealed to agent
            cookie = env.environment.user_model[i].user_state.user_cookie
            for cohort in range(NUM_COHORTS):
                # todo: simplify expressions (obvious cancellations)
                prob_cookie_given_cohort = user_cookie_cohort_dist[cookie][cohort] / user_marginal_on_cohort[cohort]
                prob_consent_given_cohort = env.environment.user_model[i].user_state.user_consent_dist[cohort]
                cohort_beliefs[i, cohort] = prob_cookie_given_cohort * prob_consent_given_cohort * user_marginal_on_cohort[cohort]
            cohort_beliefs[i, :] = cohort_beliefs[i, :] / sum(cohort_beliefs[i, :])
            # normalization = sum(user_cookie_cohort_dist[cookie])
            # cohort_beliefs[i, :] = np.array([user_cookie_cohort_dist[cookie][i] / normalization for i in range(NUM_COHORTS)])
        else:  # then cookie is not revealed to agent
            for cohort in range(NUM_COHORTS):
                prob_not_consent_given_cohort = 1 - env.environment.user_model[i].user_state.user_consent_dist[cohort]
                cohort_beliefs[i, cohort] =  prob_not_consent_given_cohort * user_marginal_on_cohort[cohort]
            cohort_beliefs[i, :] = cohort_beliefs[i, :] / sum(cohort_beliefs[i, :])

    return cohort_beliefs


def get_click_probabilities(env, topic_affinities):
    '''
    Computes probabilities of clicking on each ad given topic affinities.

    :param env: simulation environment:
    :param topic_affinities: user's topic affinities across ads
    :return: click probabilities across topics
    '''
    no_click_mass = env.environment.user_model[0].user_state.no_click_mass  # assumption: uniform no_click_mass

    probabilities_of_click = np.zeros((2, NUM_TOPICS))   # first row is probability of clicking on topic, second row is probability of not clicking
    for topic in range(NUM_TOPICS):
        expected_score = score_ad(topic_affinities[topic])
        expected_logit_scores = np.append(expected_score, no_click_mass)
        probabilities_of_click[:, topic] = expected_logit_scores / np.sum(expected_logit_scores)

    return probabilities_of_click


def update_cohort_beliefs(env, responses, prior_cohort_beliefs):
    '''
    Updates the agent's belief users' cohorts based on a set of responses.

    :param env: simulation environment
    :param responses: collection of user responses so far
    :param cohort_distributions: current cohort distributions across users (prior)
    :return: updated belief on cohorts
    '''

    posterior_cohort_beliefs = np.zeros((NUM_USERS, NUM_COHORTS))
    expected_affinities_by_topic_by_cohort = np.exp(get_affinity_means() + TOPIC_STD ** 2 / 2)
    expected_probabilities_of_click_by_cohort = np.zeros((NUM_COHORTS, NUM_TOPICS))
    for cohort in range(NUM_COHORTS):
        expected_probabilities_of_click_by_cohort[cohort, :] = get_click_probabilities(env, expected_affinities_by_topic_by_cohort[:, cohort])[0]

    for i in range(NUM_USERS):

        # Extract agent's prior cohort belief for user i
        prior_cohort_belief = prior_cohort_beliefs[i, :]

        # Extract responses for current user
        if type(responses) is dict:
            response_set_for_current_user = [resp[i] for resp in list(responses.values())]
        else:
            if type(responses[0]) is list:  # If responses are a list of lists (we have multiple rounds of responses)
                response_set_for_current_user = [resp[i] for resp in responses]
            else:   # If responses are a list of dictionaries (we have a single round of responses)
                response_set_for_current_user = [responses[i]]

        posterior_cohort_belief = np.zeros(NUM_COHORTS)
        for resp in response_set_for_current_user:

            # Form posterior probability via Bayesian update
            # For each response, compute:
            #   p(user i is in cohort | user i gives response)
            #       = p(user i gives response | user i is in cohort) * p(user i is in cohort) / p(user i gives response)
            posterior = np.zeros(NUM_COHORTS)
            if resp['click']:   # user clicked on ad
                for cohort in range(NUM_COHORTS):
                    posterior[cohort] = expected_probabilities_of_click_by_cohort[cohort, resp['topic']] * prior_cohort_belief[cohort]
            else:   # user did not click on ad
                for cohort in range(NUM_COHORTS):
                    posterior[cohort] = (1 - expected_probabilities_of_click_by_cohort[cohort, resp['topic']]) * prior_cohort_belief[cohort]
            posterior_cohort_belief = posterior / sum(posterior)

            # Assign next prior as current posterior for next response
            prior_cohort_belief = posterior_cohort_belief

        posterior_cohort_beliefs[i, :] = posterior_cohort_belief

    return posterior_cohort_beliefs


def generate_cookie_cohort_matrix(n):
    y = 1 / (n + n ** 2)
    x = 2 * y
    matrix = np.full((n, n), y)
    for i in range(n):
        matrix[i][i] = x
    return matrix
