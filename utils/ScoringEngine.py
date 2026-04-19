
from utils.ActivityClassifier import ActivityClassifier
from utils.ADMETClassifier import ADMETClassifier

class ScoringEngine:
    def __init__(self, activity_classifier_path, admet_model_path):
        """
        Args:
            activity_classifier: The loaded ActivityClassifier instance.
            admet_model_path: Path to the ADMET model.
        """
        self.activity_classifier_model = ActivityClassifier(activity_classifier_path)
        self.admet_classifier_model = ADMETClassifier(admet_model_path)

    def get_all_scores(self, z):
        """
        CRITICAL: This must accept 'z' and pass it down.
        """
        scores = {}
        scores.update(self.admet_classifier_model.classify_admet(z))
        scores['potency'] = self.activity_classifier_model.classify_activity(z) 
        
        return scores