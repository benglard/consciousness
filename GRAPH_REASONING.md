# Graph RL Reasoning for LLMs

Recent works have explored structuring the LLM generation process as a tree or a graph. Nodes represent intermediary outputs in reasoning in response to a prompt, and edges connect these steps. At inference time, some scoring network pre-trained on a text-score aligned dataset (or another instance of the same LLM) may attempt to guide the generation to explore reasoning spaces likely to produce high-score outputs.

### Method

With graph neural networks (GNNs) we can represent and act on the entire structure of this generation process, to optimize problem solving ability. A GNN could trigger/control the expansion of lines of reasoning, to which an LLM could then provide fine-grained output.

Here the reasoning graph should be latent, storing embeddings of the intermediate textual outputs, and should be dynamic, as the process evolves. The GNN should then be optimized to guide the generation with RL, with rewards coming from accuracy and efficiency in its response.

This model can be seen as hierarchical control: a GNN higher level controller and a LLM lower-level controller. Other names often suggested here are "frontal cortex" or "system 2". In a way this model can also be seen as an actor (LLM) critic (GNN) model where the critic is active and acting. Often the critic is only used at training time.

#### Algorithm

The simplest graph structure here would be a binary tree

1. Select a pre-trained LLM and sentence embedding model. Create some untrained GNN, that ends with a binary classifier on each node
2. Prompt the LLM that it should produce responses step by step
3. Given some challenge prompt, generate 2 candidates for the first step
4. Form a tree with the prompt node connected to the 2 candidates
5. Embed these 3 and store within the node
6. Run the GNN on the graph
7. The binary classifier will judge whether to expand a given leaf node
8. If expansion is selected, collect the parents of this node to form a given textual context, and generate two next step candidates from this context
9. Embed the new candidates and add the nodes/edges to the tree
10. Repeat 6-9 until some end
11. Score all the leaf nodes and propagate scores up the tree. Add efficiency score (penalize too much node expansion)
12. Train the GNN with RL

#### Extensions

- The structure could be an arbitrary graph, with adjacency re-evaluated and re-connected at each step
- A time dynamic neural net like a GRU/LSTM could optimize node embeddings at each step, rather than simply maintaining the outputs of the sentence embedding model
- After generation of an intermediary output, another instance of the LLM could generate a critique of the reasoning or of the output so far. These critiques and their embeddings could be stored on the edges of the graph. The GNN could then incorporate edge embeddings in the model
- Learned graph pruning
- Could the controller be trained with the outputs on one LLM, and still work with others?