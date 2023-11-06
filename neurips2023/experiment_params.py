from custom_utils import *


def get_experiments():

    experiments = {}

    # EXPERIMENT 1
    num_cohorts = 2
    num_topics = 200
    similarity = 0.6  # degree of similarity of affinities across cohorts
    topic_affinity_means, topic_affinity_stds = get_topic_affinity_params(num_cohorts, num_topics, similarity)
    experiments['experiment_1'] = {
        'num_users': 1000,
        'slate_size': 1,
        'num_offline_responses': 0,
        'num_candidate_ads': 50,
        'resample_documents': True,
        'seed': 232,
        'agent':
            {
                'name': 'random-multi',
                'params': None
            },
            # {
            #     'name': 'bandit-mf',
            #     'params': {
            #         'num_features': 50,  # number of features used for matrix factorization
            #         'epsilon': 0.1,  # exploration parameter for eps-greedy bandit
            #         'batch_size': 0,  # how many responses to collect from each user before update
            #     }
            # },
        'ads': {
            'num_topics': num_topics,
            'topic_distribution': {
                'type': 'uniform',
                'params': None
            }
        },
        'users': {
            'prior': [[0.4, 0.1], [0.1, 0.4]],  # size: num_cookies by num_cohorts
            'consent_rates': [0.2, 0.3],  # size: num_cohorts
            'topic_affinity_means': topic_affinity_means,
            'topic_affinity_stds': topic_affinity_stds,
            'time_budget': 1000,  # set to a high number for persistent users
            'choice_model': {
                'type': 'MultinomialLogitChoiceModel',  # as named in recsim/choice_model.py
                'choice_features': {
                    'no_click_mass': 25.0,  # as named in arg of choice_features.get() in recsim/choice_model.py
                }
            }
        }
    }

    # EXPERIMENT 2
    # ...
    # experiments['experiment_2']

    return experiments

get_experiments()

# Suggestions for generalization:
# - extend to multiple ad features, e.g.,
#     'features': ['topic', 'quality'],
#     'feature_params': {
#         'topic': {
#             'type': 'discrete',
#             'domain': num_topics,
#             'distribution': {
#                 'type': 'uniform',
#                 'params': []
#             },
#         },
#         'quality': {
#             'type': 'continuous-interval-scalar',
#             'domain': {  # for input to gym spaces
#                 'low': 0.0,
#                 'high': 1.0,
#             },
#             'distribution': {
#                 'type': 'uniform',
#                 'params': []
#             },
#         },
#     },
# - parameters of user utility should reflect full ad feature set
