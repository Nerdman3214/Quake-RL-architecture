"""Simple placeholder for event recording in Xonotic.
In a real setup this would hook into the game's telemetry API and write JSON logs.
"""

def record_event(event_type, data):
    """Append an event to a log file."""
    import json, os
    log_path = os.path.join(os.path.dirname(__file__), 'events.log')
    with open(log_path, 'a') as f:
        json.dump({'type': event_type, 'data': data}, f)
        f.write('\n')

if __name__ == '__main__':
    # Example usage
    record_event('player_kill', {'killer':'bot1','victim':'bot2','weapon':'rocket'})
    print('Event recorded.')
