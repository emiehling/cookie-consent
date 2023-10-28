import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pickle
import os

from base_functions import *

def get_statistics(user_cohorts,
                   user_opt_in_decisions,
                   user_topic_affinities,
                   cohort_beliefs_by_episode,
                   expected_affinities_by_topic_by_episode,
                   parameters,
                   numruns):

    NUM_ROUNDS = parameters['NUM_ROUNDS']
    NUM_USERS = parameters['NUM_USERS']
    NUM_COHORTS = parameters['NUM_COHORTS']
    NUM_AVERAGING_EPISODES = int(numruns)

    cohort_error_trajectories_by_episode = {m: [] for m in range(NUM_AVERAGING_EPISODES)}
    affinity_error_trajectories_by_episode = {m: [] for m in range(NUM_AVERAGING_EPISODES)}
    for m in range(NUM_AVERAGING_EPISODES):

        # Compute belief and affinity errors by user
        cohort_error_trajectories_by_user = np.zeros((NUM_USERS, NUM_ROUNDS))
        affinity_error_trajectories_by_user = np.zeros((NUM_USERS, NUM_ROUNDS))
        for n in range(NUM_ROUNDS):
            cohort_error_trajectories_by_user[:, n] = np.abs(
                user_cohorts[m, :] - np.dot(cohort_beliefs_by_episode[m][n], [0, 1]))
            affinity_error_trajectories_by_user[:, n] = np.mean(
                np.abs(user_topic_affinities[m, :, :] - expected_affinities_by_topic_by_episode[m][n]),
                axis=1)  # affinity errors for each user averaged over topics

        # Partition users by cohort and opt-in decision
        # Use partitioning to form partitioned cohort errors and affinity errors
        users_by_cohort_by_opt_in = {cohort: [] for cohort in range(NUM_COHORTS)}
        cohort_error_trajectories_by_cohort_by_opt_in = {cohort: [] for cohort in range(NUM_COHORTS)}
        affinity_error_trajectories_by_cohort_by_opt_in = {cohort: [] for cohort in range(NUM_COHORTS)}
        for cohort in range(NUM_COHORTS):
            users_current_cohort_not_opt_in = np.intersect1d(np.where(user_opt_in_decisions[m, :] == 0),
                                                             np.where(user_cohorts[m, :] == cohort))
            users_current_cohort_opt_in = np.intersect1d(np.where(user_opt_in_decisions[m, :] == 1),
                                                         np.where(user_cohorts[m, :] == cohort))
            users_by_cohort_by_opt_in[cohort] = [users_current_cohort_not_opt_in.tolist(),
                                                 users_current_cohort_opt_in.tolist()]
            cohort_error_trajectories_by_cohort_by_opt_in[cohort] = [
                cohort_error_trajectories_by_user[users_current_cohort_not_opt_in, :].tolist(),
                cohort_error_trajectories_by_user[users_current_cohort_opt_in, :].tolist()]
            affinity_error_trajectories_by_cohort_by_opt_in[cohort] = [
                affinity_error_trajectories_by_user[users_current_cohort_not_opt_in, :].tolist(),
                affinity_error_trajectories_by_user[users_current_cohort_opt_in, :].tolist()]

        cohort_error_trajectories_by_episode[m] = cohort_error_trajectories_by_cohort_by_opt_in
        affinity_error_trajectories_by_episode[m] = affinity_error_trajectories_by_cohort_by_opt_in

    # Extract number of trajectory in each episode for each (consent, cohort) pair
    trajectory_counts = np.zeros((NUM_AVERAGING_EPISODES, NUM_COHORTS, 2))
    for m in range(NUM_AVERAGING_EPISODES):
        for cohort in range(NUM_COHORTS):
            trajectory_counts[m, cohort, 0] = len(cohort_error_trajectories_by_episode[m][cohort][0])
            trajectory_counts[m, cohort, 1] = len(cohort_error_trajectories_by_episode[m][cohort][1])

    # Compute overall statistics
    cohort_error_mean_trajectory_by_cohort_by_consent = np.zeros((NUM_COHORTS, 2, NUM_ROUNDS))
    cohort_error_var_trajectory_by_cohort_by_consent = np.zeros((NUM_COHORTS, 2, NUM_ROUNDS))
    affinity_error_mean_trajectory_by_cohort_by_consent = np.zeros((NUM_COHORTS, 2, NUM_ROUNDS))
    affinity_error_var_trajectory_by_cohort_by_consent = np.zeros((NUM_COHORTS, 2, NUM_ROUNDS))
    for cohort in range(NUM_COHORTS):
        for opt_in in [0, 1]:
            for m in range(NUM_AVERAGING_EPISODES):
                # Compute means (weighted average of means, where weights are counts of trajectories)
                cohort_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :] += \
                    trajectory_counts[m, cohort, opt_in] * \
                    np.mean(cohort_error_trajectories_by_episode[m][cohort][opt_in], axis=0)
                affinity_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :] += \
                    trajectory_counts[m, cohort, opt_in] * \
                    np.mean(affinity_error_trajectories_by_episode[m][cohort][opt_in], axis=0)
                # Compute variances (weights average of variances, where weights are squared counts of trajectories)
                cohort_error_var_trajectory_by_cohort_by_consent[cohort, opt_in] += \
                    trajectory_counts[m, cohort, opt_in]**2 * \
                    np.var(cohort_error_trajectories_by_episode[m][cohort][opt_in], axis=0)
                affinity_error_var_trajectory_by_cohort_by_consent[cohort, opt_in] += \
                    trajectory_counts[m, cohort, opt_in] ** 2 * \
                    np.var(affinity_error_trajectories_by_episode[m][cohort][opt_in], axis=0)

            # Normalize means by dividing by total count in given (cohort, consent) pair
            cohort_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :] /= sum(trajectory_counts[:, cohort, opt_in])
            affinity_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :] /= sum(trajectory_counts[:, cohort, opt_in])
            # Normalize variances by dividing by squared total count in given (cohort, consent) pair
            cohort_error_var_trajectory_by_cohort_by_consent[cohort, opt_in, :] /= sum(trajectory_counts[:, cohort, opt_in])**2
            affinity_error_var_trajectory_by_cohort_by_consent[cohort, opt_in, :] /= sum(trajectory_counts[:, cohort, opt_in]) ** 2

    return cohort_error_mean_trajectory_by_cohort_by_consent, \
           cohort_error_var_trajectory_by_cohort_by_consent, \
           affinity_error_mean_trajectory_by_cohort_by_consent, \
           affinity_error_var_trajectory_by_cohort_by_consent


def load_and_plot_single_dataset(path, opt_in_rate0, opt_in_rate1, numruns):


    parameter_str = str(opt_in_rate0) + '_' + str(opt_in_rate1) + '_'
    opt_in_rates = [opt_in_rate0, opt_in_rate1]

    print(path + '_data_agg_' + parameter_str + 'avg_' + str(numruns))

    with open(path + '_data_agg_' + parameter_str + 'avg_' + str(numruns) + '/parameters.pickle', 'rb') as handle:
        parameters = pickle.load(handle)
    with open(path + '_data_agg_' + parameter_str + 'avg_' + str(numruns) + '/rs_environments.pickle', 'rb') as handle:
        rs_environments = pickle.load(handle)
    with open(path + '_data_agg_' + parameter_str + 'avg_' + str(numruns) + '/cohort_beliefs_by_episode.pickle', 'rb') as handle:
        cohort_beliefs_by_episode = pickle.load(handle)
    with open(path + '_data_agg_' + parameter_str + 'avg_' + str(numruns) + '/expected_affinities_by_topic_by_episode.pickle', 'rb') as handle:
        expected_affinities_by_topic_by_episode = pickle.load(handle)


    NUM_ROUNDS = parameters['NUM_ROUNDS']
    NUM_TOPICS = parameters['NUM_TOPICS']
    NUM_USERS = parameters['NUM_USERS']
    NUM_COHORTS = parameters['NUM_COHORTS']
    NUM_AVERAGING_EPISODES = int(numruns)

    # Extract and store user properties (for convenience)
    user_opt_in_decisions = np.zeros((NUM_AVERAGING_EPISODES, NUM_USERS))
    user_cookies = np.zeros((NUM_AVERAGING_EPISODES, NUM_USERS))
    user_cohorts = np.zeros((NUM_AVERAGING_EPISODES, NUM_USERS))
    user_topic_affinities = np.zeros((NUM_AVERAGING_EPISODES, NUM_USERS, NUM_TOPICS))
    for m in range(NUM_AVERAGING_EPISODES):
        for i in range(NUM_USERS):
            user_opt_in_decisions[m,i] = rs_environments[m].environment.user_model[i].user_state.user_opt_in
            user_cookies[m,i] = rs_environments[m].environment.user_model[i].user_state.user_cookie
            user_cohorts[m,i] = rs_environments[m].environment.user_model[i].user_state.user_cohort
            user_topic_affinities[m,i,:] = rs_environments[m].environment.user_model[i].user_state.user_topic_affinity

    cohort_error_mean_trajectory_by_cohort_by_consent, cohort_error_var_trajectory_by_cohort_by_consent, \
    affinity_error_mean_trajectory_by_cohort_by_consent, affinity_error_var_trajectory_by_cohort_by_consent = \
        get_statistics(user_cohorts, user_opt_in_decisions, user_topic_affinities,
                       cohort_beliefs_by_episode, expected_affinities_by_topic_by_episode, parameters, numruns)


    ################
    ### PLOTTING ###
    ################

    color_map_cohort = ['red', 'blue']     # one color for each cohort

    # Plot opt-in distributions of users across episodes
    fig, axs = plt.subplots(1, 2)
    for m in range(NUM_AVERAGING_EPISODES):
        axs[0].hist(user_opt_in_decisions[m, np.where(user_cohorts[m] == 0)][0], color=color_map_cohort[0], alpha=1/NUM_AVERAGING_EPISODES)    # cohort 0
        axs[1].hist(user_opt_in_decisions[m, np.where(user_cohorts[m] == 1)][0], color=color_map_cohort[1], alpha=1/NUM_AVERAGING_EPISODES)    # cohort 1
    axs[0].set_ylim([0, NUM_USERS]); axs[1].set_ylim([0, NUM_USERS])
    axs[0].set_xticks([0, 1]); axs[1].set_xticks([0, 1])
    axs[0].set_xlabel('opt-in status for cohort 0'); axs[1].set_xlabel('opt-in status for cohort 1')
    axs[0].set_ylabel('number of users'); axs[1].set_ylabel('number of users')
    plt.show()


    # Plot cohort beliefs over time (sanity check)
    m = 0; users = range(20)   # just check for a given episode and a subset of users
    fig, axs = plt.subplots(len(users), 1)
    for i in users:
        axs[i].plot(range(NUM_ROUNDS), user_cohorts[m,i] * np.ones(NUM_ROUNDS), 'b--', alpha=0.25)
        axs[i].plot([np.dot(dist[i, :], [0, 1]) for dist in list(cohort_beliefs_by_episode[m].values())], 'r')
        axs[i].set_xlim([1, NUM_ROUNDS])
        axs[i].set_ylim([-0.1, 1.1])
        if not user_opt_in_decisions[m, i]:     # user did not opt-in
            axs[i].set_facecolor('lightgrey')
    plt.show()


    # Cohort errors
    fig0 = plt.figure(figsize=(3, 3))  # initialize plot of a specific size
    for cohort in range(NUM_COHORTS):
        for opt_in in [0, 1]:
            plt.plot(range(NUM_ROUNDS), cohort_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :],
                     linewidth=2,
                     linestyle=('solid' if opt_in else 'dashed'),
                     color=color_map_cohort[cohort])
            plt.fill_between(range(NUM_ROUNDS),
                             cohort_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :] - np.sqrt(cohort_error_var_trajectory_by_cohort_by_consent[cohort, opt_in, :]),
                             cohort_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :] + np.sqrt(cohort_error_var_trajectory_by_cohort_by_consent[cohort, opt_in, :]),
                             alpha=0.125,
                             facecolor='none',
                             linewidth=0,
                             color=color_map_cohort[cohort])
    plt.title(' cohort error ')
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, NUM_ROUNDS-1])
    plt.ylim([0, 1.0])
    plt.show()


    # Affinity errors
    fig1 = plt.figure(figsize=(3, 3))  # initialize plot of a specific size
    for cohort in range(NUM_COHORTS):
        for opt_in in [0, 1]:
            plt.plot(range(NUM_ROUNDS), affinity_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :],
                     linewidth=2,
                     linestyle=('solid' if opt_in else 'dashed'),
                     color=color_map_cohort[cohort])
            plt.fill_between(range(NUM_ROUNDS),
                             affinity_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :] - np.sqrt(affinity_error_var_trajectory_by_cohort_by_consent[cohort, opt_in, :]),
                             affinity_error_mean_trajectory_by_cohort_by_consent[cohort, opt_in, :] + np.sqrt(affinity_error_var_trajectory_by_cohort_by_consent[cohort, opt_in, :]),
                             alpha=0.125,
                             facecolor='none',
                             linewidth=0,
                             color=color_map_cohort[cohort])
    plt.title(' affinity error ')
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, NUM_ROUNDS - 1])
    plt.ylim([0.3, 0.7])
    plt.show()




