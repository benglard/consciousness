# Transformer Ideas

Some small ideas for transformer architectures

#### No MLP/Residuals

One benefit of the token-wise MLP blocks and residual connections in a transformer is to maintain some independence and diversity of token representations. Otherwise a network made only of attention blocks can mix and average tokens together through layers and degrade performance. This is like applying a gaussian blur to an image over and over.

Could it be possible to remove the residual connections and the MLP blocks and retain good performance?

Try this:

The attention heads act on the values vectors which are simply a projection of the tokens incoming to a layer. The vectors are split between heads so that each head typically is multiplied with a compressed version, by a factor of 1/num heads. A head which was the identity matrix would simply pass on these vectors without mixing amongst tokens. So always add an identity head to your other computed dot-product-softmax heads.

Add a small linear layer that will compute a scalar weight for each head in a data-dependent manner. Initialize this layer so that when the model begins training, all weight is given to the identity head. In this case, the model starts off like an MLP only network, and then can train these weights to be optimal for the given data. Multiply in these weights before mixing amongst the heads.

With respect to the new identity head, this will create a data-dependent weighted bottlenecked residual connection. With respect to the other heads, it is just another weight.

Move more weights into the head mixing linear layer that occurs after applying the attention heads. The number of new weights should equal the missing weights from the MLP blocks for a good test. Make this layer non-linear now.

#### One MoE V

Form a universal transformer (one layer looped). Form the queries/keys network in usual fashion, form the values network as a mixture of experts. This implies that each time the layer iterates, different experts will be activated to produce the output. And like above, no separate MLP block is needed - token diversity is assisted via aggregation of multiple experts. Scale weights horizontally.

I like this architecture because it is essentially a feedback mechanism between one large memory system and an attentional sensing system.

#### Virtual Heads

Multi-head attention produces an image of shape `TxTxH`. You can process this image with a conv2d to create new virtual heads, which represent learned spatial and feature-wise aggregations of the qk inner product heads. If your attention graph must remain causal, you can use a causal conv1d across each row. This can be interpreted as a local graph rewiring on top of the computed attention graph.