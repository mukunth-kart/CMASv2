import logging
import torch

from Models.AdmetClassifier.train_multitask_selfies import MultiHeadADMET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADMETClassifier:
    """
    Updated Classifier for predicting ADMET properties directly from 
    the VAE Latent Space (z).
    """
    def __init__(self, model_path: str, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # EXACT Task Order from your successful training log
        self.task_names = [
            'BBBP', 'CYP1A2_inhibition', 'CYP2C19_inhibition', 'CYP2C9_inhibition', 
            'CYP3A4_inhibition', 
            'HLM_stability', 'P-gp_substrate', 'RLM_stability', 'hERG_inhibition'
        ]
        
        self.input_dim = 128 # NEW: Latent space dim, not fingerprints
        self.model = MultiHeadADMET(latent_dim=self.input_dim, num_tasks=len(self.task_names))

        # Load Checkpoint
        logger.info(f"🚀 Loading Latent-ADMET model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle the 'model_state' key from our training script
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device).eval()
        logger.info(f"✅ ADMET Classifier Ready from {model_path}")

    def classify_admet(self, z: torch.Tensor) -> dict:
        """
        Classify ADMET properties for a latent vector z.
        
        Args:
            z (torch.Tensor): Latent vector [1, 128] or [batch, 128]
        Returns:
            dict: Task names with predicted probabilities.
        """
        # Ensure z has a batch dimension
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        z = z.to(self.device)

        logits = self.model(z)  # Shape: [1, 11]
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        temperature = 2.0
        probs = torch.sigmoid(logits / temperature).squeeze(0)

        return {task: probs[i] for i, task in enumerate(self.task_names)}
    
    def get_task_probability(self, z, task_name):
        """
        Returns the differentiable probability tensor for a specific task.
        Used primarily for Gradient Ascent.
        """
        # Ensure z is on the correct device
        z = z.to(self.device)
        
        # Forward pass (Keep the graph alive!)
        logits = self.model(z)
        
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in ADMET model.")
            
        task_idx = self.task_names.index(task_name)
        
        # Select the specific head output
        target_logit = logits[:, task_idx]
        
        # Return probability (Sigmoid)
        return torch.sigmoid(target_logit)