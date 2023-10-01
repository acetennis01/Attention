# Attention
Attention Is All You Need
Description:
This repository contains that code for Attention. The Attention is a transformer for translation tasks created by the Google Research Team; this model is for English to Spanish. The current limitation for RNNs is that it is tough to train long-range dependencies. In other words, the hidden states must contain much information for the decoder layer to output the correct words. Attention fixes this issue as the decoder is equipped to look back at the hidden states of the input. This method reduces the problem of long-range dependencies. 

As outlined in the code, the Encoder and Decoder Blocks have three major parts. These are the Multi-Head Attention, Feed Forward, and Layer Normalization. The encoder takes in the positional encoding of the input embedding and plugs it into multi-head Attention connected to a feed-forward. The decoder similarly takes in the positional encoding of the output embedding and plugs it into a multi-head attention. The output of that with the encoder output is connected to another multi-head attention. This output is then connected to a feed-forward, which is finally projected. 

This model follows the architecture outlined in the original paper(https://arxiv.org/abs/1706.03762).
Followed the tutorial by Umar Jamil to code this model.
