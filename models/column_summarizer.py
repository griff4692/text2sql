import torch.nn as nn


class ColumnSummarizer(nn.Module):
    def __init__(self, embeddings):
        super(ColumnSummarizer, self).__init__()
        self.embeddings = embeddings
        embed_dim= self.embeddings.embedding_dim
        self.summarizer = nn.LSTM(input_size=embed_dim,hidden_size=embed_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, c_ids):
        """
        :param c_ids: tensor of size batch_size, max # columns in batch, max # tokens per column
        :return: column embeddings pooled across column tokens
        tensor of size batch_size, max # columns in batch, embedding_dim
        """
        bsize, max_cols, max_toks = c_ids.shape
        c_ids_flat = c_ids.view(bsize * max_cols, -1)
        c_embeds_flat = self.embeddings(c_ids_flat)
        _, (c_h_flat, _) = self.summarizer(c_embeds_flat)
        c_h_flat = c_h_flat.transpose(1, 0).contiguous()
        return c_h_flat.view(bsize, max_cols, -1)
