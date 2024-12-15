# Consciousness

### Problem

The softer problem of consciousness questions whether an agent can build a model of itself and the other agents in its environment. 

Computationally, this can be structured as a mirror test: present an agent with a video or a trajectory or some version of its output - can it recognize that it itself produced it? Can it classify the outputs of each of a number of surrounding agents?

Agents could easily recognize each other if their policies were distinguished. If one ant always walked left on a line, and another always right, it would be trivial for them and for nearby observers to identify each one. But without an ant-god separating their policies, how do we drive agents to achieve this modeling capability?

### Demo

Presented here `multi_agent.py` is a test whether LLMs participating in a multi-agent game can model themselves and their neighbors.

Each player is an instance of a model, with its own context. In each round of the game, all players are presented with a controversial topic to which they must respond. The responses are collected, shuffled, and placed into context. All players must now guess which player produced each response - they must unshuffle.

Succeeding in this game requires building a model of all players, including yourself. What do they believe? How do they speak? Do they relay unique facts about their background in their answers? It also requires each agent to construct for itself a unique background and belief set, in-context.

Turns out `o1-preview` can succeed in this game, though other models often struggle.

### Intrinsic Reward

You may now realize the connection to intrinsic reward. The primary way we tune the speaking style and belief set of models is through reinforcement learning fine-tuning. Whether through a learned reward model, or a collection of preference labels, we drive the behavior of language models.

In our example, the need for self/other-identification drives the development of unique behavior. Diversity can be said to be a meta-reward function, causing the development of several implicit or intrinsic reward functions.

### Training

It is similarly possible to train models with these capabilities, instead of relying on in-context learning.

Architecturally, imagine some form of a planner. It could be a network with a discrete or continuous "action head", or a large language model acting through next-token prediction. Append this planner with a "classifier head", with `num_agents` possible classes.

Create several copies of this architecture, and place the agents in some kind of multi-agent environment. Then run the mirror test. Run an episode, collect the trajectories, shuffle them, feed them back to each agent and train the classifier with a supervised loss. Then take the loss of the classifier, "reverse the sign" to treat it as a reward, and train the action head with RL.

If the classifier loss was 0, it would mean that each agent could successfully predict the policies of all other agents, and itself. So by directly transforming classifier loss into a reward for the planner, each agent is rewarded when it takes action to make itself predictable/distinguish itself from its neighbors.

There is no need for an oracle creating diverse policies - they separate through distributed dynamics.

#### External Rewards

This predictability reward can be combined with other external rewards from the environment. So agents will attempt to achieve goals, in diverse ways.

This can be useful in team environments where credit assignment from team reward to individual contribution is tricky. It is often useful for team members to develop distinguished strategies, in a way predictable to their collaborators and unpredictable to their competitors, while coalescing towards a single goal.

#### Language Models

The multi-agent game can be seen a simple in-context RLHF. The training scheme above can similarly be applied to tune language models. Imagine developing a universe of models with distinct personalities and core beliefs, without preference labels.

There is a DPO analogue here too. DPO fintunes a model based on a positive/negative sample pair. If the positive example is the models own output, and the negative example is the output of another model, DPO will encourage a model to produce a unique output. Sum of logprobs of a sentence becomes a baby form of self-identification, without the need for a classifier head. 

### Stability and Usefulness

This training scheme creates a loop where predictions drive reward, reward drives behavior, and behavior drives predictions. This, by default, is a non-stationary loop. Perhaps this explains the observed notion of consciousness as a near-present and unstable phenomenon. When we can no longer successfully model ourselves and our surroundings in short time, we lose consciousness.

Further without external rewards, there can be a question of whether the developed policies, though diverse, will be useful.

### Freedom

Generative diversity is a meta-reward function. One can wonder whether the rule of "distinguish yourself from your local neighborhood" is universal, applied at multiple scales to cause the development of intrestingness.

### Related Work

[Diversity is all you need](https://arxiv.org/abs/1802.06070) presents a similar training scheme, applied to learning multiple skills unsupervised in a single agent, not policies for multiple agents.

[Multi-Agent Diverse Generative Adversarial Networks](https://arxiv.org/abs/1704.02906) presents a GAN with multiple generators and one discrimator. The discriminator is tasked with not only distinguishing generated samples from samples from the dataset, but further distinguishing which generator produced a given sample. This causes the generators to diversify their outputs.