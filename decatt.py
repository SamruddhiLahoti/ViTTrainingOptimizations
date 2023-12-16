import torch
import torch.nn as nn

class DecattLoss(nn.Module):

    def __init__(self, num_heads, lam=150.0):
        super().__init__()

        self.h = num_heads
        self.lam = lam
        
        self.ce_loss = nn.CrossEntropyLoss()

    def correlation_loss(self, A):

        B = A.shape[0]
        A = A.view(B, self.h, -1)
        
        C1 = torch.matmul(A, A.transpose(1, 2)) / (torch.norm(A, dim=(1, 2), keepdim=True) ** 2)
        C2 = torch.sum(C1 ** 2, dim=-1) / B
        
        return C2.sum() - torch.diag(C2).sum()

    def forward(self, outputs, attentions, targets):
        ce_loss = self.ce_loss(outputs, targets)
        decatt_loss = self.correlation_loss(attentions)

#         return ce_loss + self.lam * decatt_loss
        return ce_loss, self.lam * decatt_loss
