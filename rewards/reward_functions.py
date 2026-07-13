"""Reward definitions for Xonotic RL.
Each function receives an event dict and returns a numeric reward.
"""

def kill_reward(event):
    if event.get('type') == 'player_kill':
        return 1.0
    return 0.0

def death_penalty(event):
    if event.get('type') == 'player_death':
        return -1.0
    return 0.0
