from base_functions import *
import random
import os
import pickle
from tqdm import tqdm

from recommender_environment import AdvertisementSampler, UserStateSampler, RSUserModel, RSUserResponse
from recsim.simulator import environment
from recsim.simulator import recsim_gym


def run_consent_experiment(consent_0, consent_1, num_averaging_episodes):

    user_consent_dist = [consent_0, consent_1]
    print(' Current consent rates: ' + str(user_consent_dist))

    ###########################################################################
    ########################## SET UP ENVIRONMENT #############################
    ###########################################################################

    NUM_COHORTS = len(user_consent_dist)
    # user_cookie_cohort_dist = generate_cookie_cohort_matrix(NUM_COHORTS)
    user_cookie_cohort_dist = [[0.4, 0.1], [0.1, 0.4]]

    # Initialize user sampler
    user_sampler = UserStateSampler(
        user_cookie_cohort_distribution=user_cookie_cohort_dist,
        user_consent_distribution=user_consent_dist,
        user_document_mean_affinity_matrix=get_affinity_means(),
        user_document_stddev_affinity_matrix=TOPIC_STD*np.ones((NUM_TOPICS, NUM_COHORTS))
    )

    # Initialize ad sampler
    topic_dist = np.random.rand(NUM_TOPICS)
    topic_dist = topic_dist / sum(topic_dist)
    ad_sampler = AdvertisementSampler(
        topic_distribution=topic_dist,
        # topic_quality_mean=0.1*np.random.rand(NUM_TOPICS),
        # topic_quality_stddev=0.1*np.random.rand(NUM_TOPICS)
    )

    ###########################################################################
    ########################### RUN ENVIRONMENT ###############################
    ###########################################################################

    num_initial_responses_per_user = NUM_INITIAL_RESPONSES

    # constants
    expected_affinities_by_topic_by_cohort = np.exp(get_affinity_means() + TOPIC_STD ** 2 / 2)

    # Simulate the recommender process multiple times to get distributional information
    rs_environments = {}
    episode_statistics = {}
    for m in tqdm(range(num_averaging_episodes)):

        user_models = []
        for _ in range(NUM_USERS):
            user_models.append(RSUserModel(slate_size=1,
                                           user_state_ctor=user_sampler,
                                           response_model_ctor=RSUserResponse))
        raw_rs_env_multi = environment.MultiUserEnvironment(user_models,
                                                            document_sampler=ad_sampler,
                                                            num_candidates=NUM_CANDIDATES,
                                                            slate_size=1,
                                                            resample_documents=True)

        # Create RS gym environment by specifying agent reward function, store as m'th environment
        rs_environments[m] = recsim_gym.RecSimGymEnv(
            raw_environment=raw_rs_env_multi,
            reward_aggregator=agent_reward_function
        )

        # Extract agent's action and observation spaces
        action_space = rs_environments[m].action_space
        observation_space = rs_environments[m].observation_space

        # Dictionaries to track variables over episode
        recommended_ads = {n: np.zeros(NUM_USERS, dtype=int) for n in range(NUM_ROUNDS)}  # ad indices that were recommended
        recommended_topics = {n: np.zeros(NUM_USERS, dtype=int) for n in range(NUM_ROUNDS)}  # ad topics that were recommended
        ad_pool_topics = {n: np.zeros(NUM_CANDIDATES) for n in range(NUM_ROUNDS)}  # current ad pool topics
        cohort_beliefs = {n: np.zeros((NUM_USERS, NUM_COHORTS)) for n in range(NUM_ROUNDS)}  # expected user affinities
        expected_affinities_by_topic = {n: np.zeros((NUM_USERS, NUM_TOPICS)) for n in range(NUM_ROUNDS)}  # expected user affinities by topic
        expected_click_probabilities_by_topic = {n: np.zeros((NUM_USERS, NUM_TOPICS)) for n in range(NUM_ROUNDS)}  #
        user_factor_matrices = {n: np.zeros((NUM_USERS, NUM_FEATURES)) for n in range(NUM_ROUNDS)}  # user factor matrices
        ad_factor_matrices = {n: np.zeros((NUM_TOPICS, NUM_FEATURES)) for n in range(NUM_ROUNDS)}  # ad factor matrices
        observations = {n: [] for n in range(NUM_ROUNDS)}  # response tuples
        user_responses = {n: [] for n in range(NUM_ROUNDS)}  # user responses
        rewards = {n: [] for n in range(NUM_ROUNDS)}

        # Enumerate over interaction rounds
        for n in tqdm(range(NUM_ROUNDS)):

           # If first round then initialize ad pool, form beliefs on cohorts to generate expected affinities, estimate latent MF factors, and generate recommendations
            if n == 0:

                # Initialize environment and define the candidate ads (ad pool)
                observations[0] = rs_environments[m].reset()
                ad_pool_topics[0] = random.sample(range(NUM_TOPICS), NUM_CANDIDATES)

                # Form cohort beliefs based on opt-in decisions and initial user responses
                prior_cohort_beliefs = get_cohort_prior_beliefs(rs_environments[m]) # form prior beliefs on cohorts due to opt-in decisions and cookie values (if opt-in)
                user_responses[0] = sample_user_responses(rs_environments[m], action_space, num_initial_responses_per_user) # collect initial pool of user responses
                cohort_beliefs[0] = update_cohort_beliefs(rs_environments[m], user_responses[0], prior_cohort_beliefs)  # update cohort distributions based on user responses
                cohort_beliefs[0] = prior_cohort_beliefs

                # Form expected affinities from cohort distributions, estimate latent MF factors, and generate recommendations
                for i in range(NUM_USERS):
                    expected_affinities_by_topic[0][i, :] = np.dot(cohort_beliefs[0][i, :], expected_affinities_by_topic_by_cohort.T)
                    expected_click_probabilities_by_topic[0][i, :] = get_click_probabilities(rs_environments[m], expected_affinities_by_topic[0][i, :])[0]
                recommended_ads[0] = action_space.sample()
                recommended_topics[0] = [ad_pool_topics[0][ad[0]] for ad in recommended_ads[0]]

            else:

                # Update factor matrix estimates (every SIZE_BATCH steps)
                if not n % SIZE_BATCH:

                    # Update beliefs based on new user responses, use to compute expected affinities, form expected click probabilities from expected affinities
                    new_user_responses = [user_responses[k] for k in range(n - SIZE_BATCH + 1, n + 1)]
                    cohort_beliefs[n] = update_cohort_beliefs(rs_environments[m], new_user_responses, cohort_beliefs[n - 1])
                    for i in range(NUM_USERS):
                        expected_affinities_by_topic[n][i, :] = np.dot(cohort_beliefs[n][i, :], expected_affinities_by_topic_by_cohort.T)
                        expected_click_probabilities_by_topic[n][i, :] = get_click_probabilities(rs_environments[m], expected_affinities_by_topic[n][i, :])[0]

                    # Update estimates using whole history of user responses (initial offline responses and current online response history) and impressions
                    if num_initial_responses_per_user > 0:
                        initial_response_pool = [user_responses[0][nr] for nr in range(num_initial_responses_per_user)]
                        history_user_responses = initial_response_pool + [user_responses[k] for k in range(1, n+1)]
                    else:
                        history_user_responses = [user_responses[k] for k in range(1,n+1)]
                    user_factor_matrices[n], ad_factor_matrices[n], training_errors = get_estimated_factors(history_user_responses, expected_click_probabilities_by_topic[n])

                    # print(training_errors[-1])
                    # total_training_errors[n] = training_errors

                else:  # If not a batch update step, then just copy previous values

                    cohort_beliefs[n] = cohort_beliefs[n-1]
                    expected_affinities_by_topic[n] = expected_affinities_by_topic[n-1]
                    expected_click_probabilities_by_topic[n] = expected_click_probabilities_by_topic[n-1]
                    user_factor_matrices[n], ad_factor_matrices[n] = user_factor_matrices[n-1], ad_factor_matrices[n-1]

                # Generate recommendations (and store recommended topics by mapping to current ad pool)
                recommended_ads[n] = get_slates(user_factor_matrices[n], ad_factor_matrices[n], ad_pool_topics[n])
                recommended_topics[n] = [ad_pool_topics[n][ad[0]] for ad in recommended_ads[n]]

            # Generate observations and responses via environment's step method on the current recommended topics
            observations[n], rewards[n], _, _ = rs_environments[m].step(recommended_topics[n])

            # Define user responses (for belief update and latent factor estimation)
            user_responses[n+1] = [obs[0] for obs in observations[n]['response']]

            # Define updated ad pool for next round
            ad_pool_topics[n+1] = random.sample(range(NUM_TOPICS), NUM_CANDIDATES)

        # Store statistics
        episode_statistics[m] = {'cohort_beliefs': cohort_beliefs,
                                 'ad_pool_topics': ad_pool_topics,
                                 'expected_affinities_by_topic': expected_affinities_by_topic,
                                 'expected_click_probabilities_by_topic': expected_click_probabilities_by_topic,
                                 'user_factor_matrices': user_factor_matrices,
                                 'ad_factor_matrices': ad_factor_matrices,
                                 'recommended_topics': recommended_topics,
                                 'observations': observations
                                 }



    ###########################################################################
    ######################### PICKLE DATA  ####################################
    ###########################################################################

    parameters = {  'NUM_CANDIDATES': NUM_CANDIDATES,
                    'NUM_ROUNDS': NUM_ROUNDS,
                    'NUM_TOPICS': NUM_TOPICS,
                    'NUM_USERS': NUM_USERS,
                    'NUM_COHORTS': NUM_COHORTS,
                    'NUM_AVERAGING_EPISODES': num_averaging_episodes,
                    'NUM_FEATURES': NUM_FEATURES,
                    'SIZE_BATCH': SIZE_BATCH,
                    'NUM_INITIAL_RESPONSES': NUM_INITIAL_RESPONSES,
                    'user_cookie_cohort_dist': user_cookie_cohort_dist,
                    'user_consent_dist': user_consent_dist,
                    'TOPIC_MEANS': TOPIC_MEANS,
                    'TOPIC_STD': TOPIC_STD
                    }

    path = '/dccstor/autofair/rs_experiments/data/data_new_' + \
           '_'.join(map(str, user_consent_dist)) + \
           '_' + str(NUM_ROUNDS) + \
           '_avg_' + str(num_averaging_episodes) + \
           str(np.random.choice(range(1000, 9999)))
    path = 'data/data_' + \
           '_'.join(map(str, user_consent_dist)) + \
           '_' + str(NUM_ROUNDS) + \
           '_avg_' + str(num_averaging_episodes)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    # append_str = ''
    with open(path + '/parameters.pickle', 'wb') as handle:
        pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + '/rs_environments.pickle', 'wb') as handle:
        pickle.dump(rs_environments, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + '/episode_statistics.pickle', 'wb') as handle:
        pickle.dump(episode_statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(path + '/cohort_beliefs_by_episode.pickle', 'wb') as handle:
    #     pickle.dump(cohort_beliefs_by_episode, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(path + '/expected_affinities_by_topic_by_episode.pickle', 'wb') as handle:
    #     pickle.dump(expected_affinities_by_topic_by_episode, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(path + '/observations_by_episode.pickle', 'wb') as handle:
    #     pickle.dump(observations_by_episode, handle, protocol=pickle.HIGHEST_PROTOCOL)


