import torch.nn as nn


class QuestionSummarizer(nn.Module):
    def __init__(self, embeddings):
        super(QuestionSummarizer, self).__init__()
        self.embeddings = embeddings
        embed_dim= self.embeddings.embedding_dim
        self.summarizer = nn.LSTM(
            input_size=embed_dim, hidden_size=embed_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, q_ids):
        """
        :param q_ids: tensor of size batch_size, max # question tokens
        :return: summarize question with an LSTM (for now)
        tensor of size batch_size, embedding_dim
        """
        bsize, max_toks = q_ids.shape
        q_embeds = self.embeddings(q_ids)

        q_h, _ = self.summarizer(q_embeds)
        return q_h
        # _, (q_summary, _) = self.summarizer(q_embeds)
        # return q_summary.transpose(1, 0).contiguous().view(bsize, -1)
