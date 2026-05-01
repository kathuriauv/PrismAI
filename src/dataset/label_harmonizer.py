
EMOTION_TO_ID = {
    "neutral": 0,
    "anger": 1,
    "joy": 2,      # Joy has both 'happy' and 'excited'
    "sadness": 3
}

ID_TO_EMOTION = {v: k for k, v in EMOTION_TO_ID.items()}

def map_iemocap_label(raw_label: str):
    """
    IEMOCAP labels are short codes: 'neu', 'ang', 'hap', 'exc', 'sad', 'fru', 'sur', 'fea'.
    We merge 'hap' and 'exc' into 'joy'. We discard the rest.
    """
    mapping = {
        'neu': 'neutral',
        'ang': 'anger',
        'hap': 'joy',
        'exc': 'joy',
        'sad': 'sadness'
    }
    
    clean_label = mapping.get(raw_label.lower())
    if clean_label:
        return EMOTION_TO_ID[clean_label]
    return None  # Returns None for discarded emotions like frustration/surprise

def map_meld_label(raw_label: str):
    """
    MELD labels are full words: 'neutral', 'anger', 'joy', 'sadness', 'surprise', 'fear', 'disgust'.
    """
    mapping = {
        'neutral': 'neutral',
        'anger': 'anger',
        'joy': 'joy',
        'sadness': 'sadness'
    }
    
    clean_label = mapping.get(raw_label.lower())
    if clean_label:
        return EMOTION_TO_ID[clean_label]
    return None  # Returns None for discarded emotions like disgust/fear

if __name__ == "__main__":
    # TEST BLOCK 
    print("Harmonizer Test ")
    print(f"IEMOCAP 'exc' maps to ID: {map_iemocap_label('exc')} ({ID_TO_EMOTION[map_iemocap_label('exc')]})")
    print(f"MELD 'joy' maps to ID: {map_meld_label('joy')} ({ID_TO_EMOTION[map_meld_label('joy')]})")
    print(f"IEMOCAP 'fru' (Frustration) maps to: {map_iemocap_label('fru')} (Discarded)")
    print("Over.")