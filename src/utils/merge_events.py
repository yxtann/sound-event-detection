import json
from typing import List, Dict, Any

def merge_close_events(events: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Merge events with the same label if the gap between previous offset and next onset is <= threshold.
    
    Args:
        events: List of event dictionaries with 'event_onset', 'event_offset', 'event_label', and 'file'
        threshold: Maximum gap (in seconds) between events to merge them
    
    Returns:
        List of merged events
    """
    if not events:
        return []
    
    # Sort events by onset time
    sorted_events = sorted(events, key=lambda x: x['event_onset'])
    
    # Group events by label
    events_by_label = {}
    for event in sorted_events:
        label = event['event_label']
        if label not in events_by_label:
            events_by_label[label] = []
        events_by_label[label].append(event)
    
    # Merge events for each label
    merged_events = []
    for label, label_events in events_by_label.items():
        if not label_events:
            continue
        
        # Process events of the same label
        current_merged = [label_events[0].copy()]
        
        for i in range(1, len(label_events)):
            prev_event = current_merged[-1]
            next_event = label_events[i]
            
            # Calculate gap between previous offset and next onset
            gap = next_event['event_onset'] - prev_event['event_offset']
            
            if gap <= threshold:
                # Merge: extend the offset to the next event's offset
                current_merged[-1]['event_offset'] = max(
                    current_merged[-1]['event_offset'],
                    next_event['event_offset']
                )
            else:
                # Gap is too large, start a new merged event
                current_merged.append(next_event.copy())
        
        merged_events.extend(current_merged)
    
    # Sort merged events by onset time (since we processed by label)
    merged_events.sort(key=lambda x: x['event_onset'])
    
    return merged_events


def process_json_file(input_file: str, output_file: str, threshold: float = 0.5):
    """
    Process the JSON file and merge close events.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        threshold: Maximum gap (in seconds) between events to merge them
    """
    # Load the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each file's events
    merged_data = {}
    for filename, events in data.items():
        merged_events = merge_close_events(events, threshold)
        merged_data[filename] = merged_events
        print(f"Processed {filename}: {len(events)} -> {len(merged_events)} events")
    
    # Save the merged results
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"\nMerged results saved to {output_file}")


if __name__ == "__main__":
    import sys
    
    # Usage: python merge_events.py <threshold>
    # Example: python merge_events.py 0.5
    
    # Default threshold (0.5 seconds)
    threshold = 0.5
    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
    
    input_file = "crnn_results.json"
    output_file = "crnn_results_merged.json"
    
    print(f"Merging events with threshold: {threshold} seconds")
    process_json_file(input_file, output_file, threshold)

