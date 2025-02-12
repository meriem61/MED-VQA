from FusionNetwork import MultiHeadAttentionFusionNetwork
from Classifier_model  import Classifier
from torch import nn

class VQAClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_heads=16 ,dropout = 0.1, num_classes=2):
        super(VQAClassifier, self).__init__()
        self.fusion = MultiHeadAttentionFusionNetwork(embed_dim, num_heads)
        self.classifier = Classifier(num_classes=num_classes, dropout_prob=dropout)
    
    def forward(self, image_emb, text_emb):
        fused_representation = self.fusion(image_emb, text_emb).squeeze(1)
        output = self.classifier(fused_representation)
        return output   