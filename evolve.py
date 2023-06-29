from rules import *

#### Probably useless function ####

def evolve(state : np.array) -> np.array:
    next_state = apply_rules(state)
    return next_state
