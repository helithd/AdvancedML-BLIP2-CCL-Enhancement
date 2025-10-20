import torch
import torch.nn as nn
import torch.nn.functional as F
# Note: For real implementation, you would need to install the Hugging Face library: pip install transformers
# The actual class you would modify deeply is Blip2QFormer, but for a clean
# demonstration, we subclass the main model and integrate the loss here.
from transformers import Blip2ForConditionalGeneration 

class Blip2WithCCL(Blip2ForConditionalGeneration):
    """
    BLIP-2 model enhanced with the Contrastive Consistency Loss (CCL)
    applied to the Querying Transformer (Q-Former) output during representation learning (Stage 1).
    """
    def __init__(self, config):
        super().__init__(config)
        self.lambda_ccl = 0.1  # Weight for the CCL, as defined in the paper
        self.temperature = 0.07 # Standard temperature for contrastive loss (tau)

    def contrastive_consistency_loss(self, query_tokens, text_embeds, text_embeds_batch):
        """
        Calculates the Contrastive Consistency Loss (CCL) = L_QAC + L_QTC.
        
        Args:
            query_tokens (torch.Tensor): Output tokens from the Q-Former (B, N_Q, D).
            text_embeds (torch.Tensor): The positive text embedding (B, 1, D).
            text_embeds_batch (torch.Tensor): Batch of text embeddings (B, 1, D) for the negative pool.
        
        Returns:
            torch.Tensor: The calculated CCL value.
        """
        B, N_Q, D = query_tokens.shape
        
        # --- 1. L_QAC (Query-Aggregate Consistency Term) ---
        # Z_agg (B, 1, D)
        Z_agg = torch.mean(query_tokens, dim=1, keepdim=True)
        
        # Sim between each query token and its aggregate (B, N_Q)
        sim_q_agg = F.cosine_similarity(query_tokens, Z_agg, dim=-1) 
        
        # L_QAC: minimize (1 - similarity) across all tokens and batches
        L_QAC = torch.mean(1.0 - sim_q_agg) 

        # --- 2. L_QTC (Query-Text Consistency Term - InfoNCE) ---
        
        # Flatten queries: (B * N_Q, D)
        queries_flat = query_tokens.reshape(-1, D)
        
        # Flatten text embeddings (the pool of all positives/negatives in batch): (B, D)
        # Note: We use the positive text embeds as both positive and in the negative pool for demonstration
        text_pool = text_embeds_batch.squeeze(1) # (B, D)
        
        # Full similarity matrix: Sim(Query, Text_Pool) -> (B * N_Q, B)
        # Each entry (i, j) is the similarity between the i-th query token and the j-th text embedding.
        sim_matrix = F.cosine_similarity(queries_flat.unsqueeze(1), text_pool.unsqueeze(0), dim=2) / self.temperature
        
        # Create target indices: For the i-th query token (which comes from image B'), 
        # the positive text embedding is the one at index B'.
        # Since queries_flat has a block structure (N_Q tokens for B=0, N_Q tokens for B=1, etc.),
        # the positive text index repeats N_Q times.
        target = torch.arange(B, device=queries_flat.device).repeat_interleave(N_Q) # (B * N_Q)

        # L_QTC is the standard Cross-Entropy loss on the similarity matrix
        # This maximizes the similarity of (z_i, t_positive) and minimizes (z_i, t_negative)
        L_QTC = F.cross_entropy(sim_matrix, target)
        
        # Final CCL is a weighted sum
        L_CCL = L_QAC + L_QTC
        
        return L_CCL

    def forward(self, pixel_values, text_input_ids, attention_mask, **kwargs):
        """
        Modified forward pass to calculate and add the CCL during the
        Vision-Language Representation Learning stage (Stage 1).
        
        This logic is a simplified representation of how custom loss is added.
        """
        # --- 1. Original BLIP-2 Forward Pass (Simulated) ---
        # We assume the base model calculates the original loss (L_BLIP2)
        # and returns the required Q-Former and text embeddings.
        
        # In a real implementation, you would call the base forward and extract:
        # 1. outputs.loss (L_BLIP2)
        # 2. qformer_output_tokens (Z)
        # 3. text_embeddings (t_positive and t_batch for the negative pool)
        
        # --- Placeholder Values for Demonstration ---
        batch_size = pixel_values.shape[0]
        num_queries = self.config.qformer_config.num_query_tokens # e.g., 32
        hidden_dim = self.config.qformer_config.hidden_size # e.g., 768
        
        # Simulate extracted components from the Q-Former forward pass
        original_loss = torch.rand(1, device=pixel_values.device, requires_grad=True) # Fake L_BLIP2
        
        # Simulated Q-Former Output Z (B, N_Q, D)
        # Using random data for the demo
        query_tokens = torch.randn(batch_size, num_queries, hidden_dim, device=pixel_values.device) 
        
        # Simulated Text Embeddings t (B, 1, D)
        text_embeds = torch.randn(batch_size, 1, hidden_dim, device=pixel_values.device)
        text_embeds_batch = text_embeds.clone() # Use positive embeds as the batch pool for demo
        
        # --- 2. Calculate CCL ---
        L_CCL = self.contrastive_consistency_loss(query_tokens, text_embeds, text_embeds_batch)
        
        # --- 3. Combine Loss ---
        total_loss = original_loss + self.lambda_ccl * L_CCL
        
        # The final dictionary mimics the output structure of the base Hugging Face model
        return {"loss": total_loss, "original_loss": original_loss, "ccl_loss": L_CCL}

# NOTE: The actual forward method in the base class (Blip2ForConditionalGeneration)
# handles both Stage 1 and Stage 2. A proper research implementation would require
# creating a dedicated training script that calls different loss functions
# based on the pre-training stage (V-L RL vs V-L GL).

if __name__ == '__main__':
    # Simple test of the loss calculation
    print("Testing CCL Loss calculation...")
    # Create a mock configuration with necessary parameters
    class MockConfig:
        def __init__(self):
            self.qformer_config = type('QConfig', (object,), {'num_query_tokens': 32, 'hidden_size': 768})
            self.model_type = "blip2" # Needed by from_pretrained, but we won't load weights
            
    model = Blip2WithCCL(MockConfig())
    # Mock data batch
    mock_pixel_values = torch.zeros(4, 3, 224, 224) # Batch of 4 images
    mock_text_ids = torch.zeros(4, 10, dtype=torch.long)
    mock_attention_mask = torch.ones(4, 10, dtype=torch.long)
    
    outputs = model.forward(mock_pixel_values, mock_text_ids, mock_attention_mask)
    
    print(f"Total Loss (L_BLIP2 + lambda * L_CCL): {outputs['loss'].item():.4f}")
    print(f"L_QAC + L_QTC (CCL): {outputs['ccl_loss'].item():.4f}")