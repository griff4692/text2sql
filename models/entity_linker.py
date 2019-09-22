import numpy as np
import torch
import torch.nn as nn

from models.column_summarizer import ColumnSummarizer
from models.question_summarizer import QuestionSummarizer


class EntityLinker(nn.Module):
    def __init__(self, embed_matrix=None):
        super(EntityLinker, self).__init__()
        embed_dim = embed_matrix.shape[1]
        self.embeddings = nn.Embedding(embed_matrix.shape[0], embed_matrix.shape[1], padding_idx=0)
        self.embeddings.load_state_dict({'weight': torch.from_numpy(embed_matrix)})
        self.embeddings.weight.requires_grad = False

        self.column_summarizer = ColumnSummarizer(embeddings=self.embeddings)
        self.question_summarizer = QuestionSummarizer(embeddings=self.embeddings)

        self.hidden_layer = nn.Linear(embed_dim * 5, embed_dim)
        self.output = nn.Linear(embed_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def compute_att(self, context, query, context_lens, query_lens):
        cq_sim = torch.bmm(context, query.transpose(2, 1)) / np.sqrt(self.embeddings.embedding_dim)

        mask = self.compute_sim_mask(cq_sim.size()[0], cq_sim.size()[1], cq_sim.size()[2], context_lens, query_lens)
        cq_sim.masked_fill(mask, float('-inf'))
        q_att_weights = self.softmax(cq_sim)
        weighted_q = torch.bmm(q_att_weights, query)
        return weighted_q

    def compute_sim_mask(self, batch_size, max_row_size, max_col_size, row_lens, col_lens):
        mask = torch.BoolTensor(batch_size, max_row_size, max_col_size)
        mask.fill_(0)
        for bidx in range(batch_size):
            row_len = row_lens[bidx]
            col_len = col_lens[bidx]
            if row_len < max_row_size:
                mask[bidx, row_len:, :] = 1
            if col_len < max_col_size:
                mask[bidx, :, col_len:] = 1
        return mask

    def forward(self, X):
        (q_ids, c_ids, num_qs, num_cols) = X
        c_h = self.column_summarizer(c_ids)
        q_h, q_summary = self.question_summarizer(q_ids)
        weighted_q = self.compute_att(c_h, q_h, num_cols, num_qs)
        q_summary_tiled = q_summary.unsqueeze(1).repeat(1, c_h.size()[1], 1)
        q_aware_c_h = torch.cat(
            [q_summary_tiled, c_h, weighted_q, c_h * weighted_q, torch.abs(c_h - weighted_q)], dim=-1)
        return self.output(self.tanh(self.hidden_layer(q_aware_c_h)))
