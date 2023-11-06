# 

This repo contains a simulation framework for studying the impact of cookie consent on recommendation dynamics. The 
code is based on the `RecSim` package (see: https://github.com/google-research/recsim). We have developed an additional environment, see 
`custom_environments/recommender_environment.py`, that integrates with `RecSim`, for the purposes of studying the impact 
of user consent on the recommender dynamics.





### Python instructions

Our simulator is based on RecSim and a fork of scikit-learn (that modifies NMF to handle missing values). We recommend 
creating a virtual environment. Once within the virtual environment, navigate to the home directory and run the 
following from a terminal to install: 



```shell 
bash setup.sh
```


### Running


Parameters of the simulation environment are specified in `experiment_params.py`. A sample parameter set is shown below:

```
num_cohorts = 2
num_topics = 2000
similarity = 0.6  # degree of similarity of affinities across cohorts
topic_affinity_means, topic_affinity_stds = get_topic_affinity_params(num_cohorts, num_topics, similarity)
experiments['experiment_1'] = {
    'num_users': 10000,
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
        'time_budget': 1000,
        'choice_model': {
            'type': 'MultinomialLogitChoiceModel',  # as named in recsim/choice_model.py
            'choice_features': {
                'no_click_mass': 25.0,  # as named in arg of choice_features.get() in recsim/choice_model.py
            }
        }
    }
}
```

### Bibtex

If you use the recommender environment in your own research, please cite the paper below in which it was published.

```
@article{miehling2023cookie,
    title={Cookie Consent Has Disparate Impact on Estimation Accuracy},
    author={Erik Miehling and Rahul Nair and Elizabeth M. Daly and Karthikeyan Natesan Ramamurthy and Robert Nelson Redmond},
    journal={Advances in Neural Information Processing Systems},
    year={2023}
}
```


