from run_consent import *
import sys

consent_0 = float(sys.argv[1])
consent_1 = float(sys.argv[2])
num_episodes = int(sys.argv[3])

run_consent_experiment(consent_0, consent_1, num_episodes)