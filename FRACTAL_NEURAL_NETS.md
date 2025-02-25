# Fractal Neural Networks

### Background

Much value would be derived from smaller neural networks achieving performance similar to larger ones. A few strategies are currently under exploration to scale the usable compute of a set of neural network layers/trainable parameters. These include:
* Scaling token use with chain of thought
* Scaling layers by iterating/repeating layers

In nature we often find evolution generating fractals in order to increase the effective utilization of a compact set of resources. It is natural, therefore, to wonder whether this principle is/was taken advantage of in the development of brains. The human visual cortex appears to only have 6 distinct "layers" of computation. This minimal depth likely suggests that a much greater level of parallelism, complexity, and feedback is employed within each layer, compared to our current artifical networks.

### Fractal Neural Networks

Let us consider each layer in a neural network as a node in a graph. This graph is a computational graph - where node application represents running a function. The connectivity between layers can be described with an adjacency matrix. Interestingly, one can grow a graph with a self-similar structure by recurrently calculating the kronecker product (aka. tensor product/matrix outer product) of its adjacency matrix with itself.

Taking the tensor product of two graphs creates a new node set equal to the cartesian product of the nodes. Since our nodes are functions, each node stores the cartesian product of two functions. Edges are created in the larger graph if the corresponding nodes were previously connected in the original graphs.

Consider a network with the following adjacency:

```
[[1, 0, 0]
 [1, 1, 0]
 [1, 1, 1]]
```

For representation of neural nets, we can think of the self-connections on the diagonal as running the layer. The network is run row by row. Any edges before the diagonal represent inputs to the layer. So this is a 3-layer network, where data flows like:

```
X1 = Layer1(input)
X2 = Layer2(X1)
X3 = Layer3(X1 + X2)
```

You could further imagine weighting these edges, so that for instance ```X3 = Layer3(w1 * X1 + w2 * X2)```. These weights could either be learnt once statically, or be made dynamic and token-wise like modern mixture-of-experts layers.

"kroneckering" this matrix with itself creates the 9x9 matrix (which begins to resemble Pascal's triangle)

```
[[1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 1, 0, 0, 0, 0, 0],
 [1, 1, 0, 1, 1, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 0, 0, 0],
 [1, 0, 0, 1, 0, 0, 1, 0, 0],
 [1, 1, 0, 1, 1, 0, 1, 1, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

Each node is now a composite representing the application of two functions. I find it easier to visualize the data flow with strings. So we will name each layer as `a, b, c`. The kronecker product is

```
[['aa', '',   '',   '',   '',   '',   '',   '',   ''],
 ['aa', 'ab', '',   '',   '',   '',   '',   '',   ''],
 ['aa', 'ab', 'ac', '',   '',   '',   '',   '',   ''],
 ['aa', '',   '',   'ba', '',   '',   '',   '',   ''],
 ['aa', 'ab', '',   'ba', 'bb', '',   '',   '',   ''],
 ['aa', 'ab', 'ac', 'ba', 'bb', 'bc', '',   '',   ''],
 ['aa', '',   '',   'ba', '',   '',   'ca', '',   ''],
 ['aa', 'ab', '',   'ba', 'bb', '',   'ca', 'cb', ''],
 ['aa', 'ab', 'ac', 'ba', 'bb', 'bc', 'ca', 'cb', 'cc']]
```

The data flow now has two parallel paths along the diagonal:

```
X1 = (a(input), a(input))
X2 = (a(X1), b(X1))
X3 = (a(X1 + X2), c(X1 + x2))
...
```

By continuing to follow the diagonal you can see that path 1 will flow like aaabbbccc, and path 2 will flow like abcabcabc. But with different inputs at each step created via interesting new skip connections. So from a set of 3 layers we have created effectively 9 + 9 = 18 computational steps. Even just considering one path means we have squared the usable compute of a fixed set of layers in a principled fashion.

You could continue "kroneckering" this adjancency to create larger structures from the same minimal set of original nodes. This would create an increased level of recursion and parameter re-use.

If you had two or more separate graphs of 3 layers each, you could merge them also with this structure (resembling a multi-fractal). Perhaps the brain utilizes a similar structure to grow networks from a small set of indepedent component experts. This could potentially be useful for multi-task learning or multi-data center distributed learning.

A final possibility is to simply arrange 9 distinct layers with the given connectivity. This network would be effectively more parallel and effectively less deep, but would not be an iterated function sytem, nor posess any less parameters. It would essentially be a constrained version of a densely connected network of 9 layers. Instead of growing dense networks on a line, we'd be growing sparse networks on a triangle or a honeycomb for instance.

### Transformers

The uniform layer structure of transformers meshes well with this framework. Since each self-attention layer will usually have receptive field equal to the entire context, we don't have to be concerned about the sequence of layers determing receptive field and therby impacting performance, like we would with convnets.

With modern LLMs, it appears we often struggle to effectively train the deepest layers, and middle layers often struggle to effectively distinguish themselves. Could a shallower network with increased parallelism and new skip connections help solve these problems? Could we learn the optimal base fractal structure? Could we learn to vary the fractal scale at inference time to mimic latent adaptive reasoning?

### Aside

How to take the kronecker product of matrices of strings:

```python
def kron_str(a, b):
    c = np.core.defchararray.add(a[:,None,:,None], b[None,:,None,:])
    return c.reshape(a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])
>>> a = np.array([['a','b'],['c','d']])
>>> b = np.array([['e','f'],['g','h']])
>>> kron_str(a,b)
array([['ae', 'af', 'be', 'bf'],
       ['ag', 'ah', 'bg', 'bh'],
       ['ce', 'cf', 'de', 'df'],
       ['cg', 'ch', 'dg', 'dh']], dtype='<U2')
```

### References

[1] For a general overview of kronecker graphs: [https://cs.stanford.edu/people/jure/pubs/kronecker-jmlr10.pdf](https://cs.stanford.edu/people/jure/pubs/kronecker-jmlr10.pdf)
[2] Implications for cognition: [https://arxiv.org/abs/1402.7038](https://arxiv.org/abs/1402.7038)
[3] [DenseFormer](https://arxiv.org/abs/2402.02622)
[4] [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887v1)