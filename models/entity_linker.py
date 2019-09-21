import torch
import torch.nn as nn

from models.column_summarizer import ColumnSummarizer
from models.question_summarizer import QuestionSummarizer


class EntityLinker(nn.Module):
    def __init__(self, embed_matrix=None, vocab=None):
        super(EntityLinker, self).__init__()
        embed_dim = embed_matrix.shape[1]
        self.embeddings = nn.Embedding(embed_matrix.shape[0], embed_matrix.shape[1], padding_idx=0)
        self.embeddings.load_state_dict({'weight': torch.from_numpy(embed_matrix)})
        self.embeddings.weight.requires_grad = False

        self.column_summarizer = ColumnSummarizer(embeddings=self.embeddings)
        self.question_summarizer = QuestionSummarizer(embeddings=self.embeddings)

        self.output = nn.Linear(embed_dim * 2, 1)

    def forward(self, X):
        (q_ids, c_ids) = X
        c_h = self.column_summarizer(c_ids)
        q_summary = self.question_summarizer(q_ids)
        q_summary_tiled = q_summary.unsqueeze(1).repeat(1, c_h.size()[1], 1)
        q_aware_c_h = torch.cat([c_h, q_summary_tiled], dim=-1)
        return self.output(q_aware_c_h)
