# Environment parameters

NUM_CANDIDATES = 50     # size of ad pool in each round
NUM_ROUNDS = 50         # length of the episode
NUM_TOPICS = 200        # number of unique topics
NUM_USERS = 200         # number of users (fixed set, no user resampling)
NUM_COHORTS = 2
SIZE_BATCH = 2
NUM_INITIAL_RESPONSES = 0

NUM_FEATURES = 50       # number of latent features to describe users and ads

# joint distribution on cookies and cohorts, must have NUM_COHORTS columns
user_cookie_cohort_dist = [[0.4, 0.1],
                           [0.1, 0.4]]

# parameters for synthetically generated topic affinities
TOPIC_MEANS = [0.3, 0.7]
TOPIC_STD = 0.3
