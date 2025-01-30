# LLM Safety

### Analogy

If a self-driving car always turned right at an intersection, its behavior would be predictable to surrounding agents. Nearby cars, pedestrians, and bicyclists could modify their own plans in accordance with their predictions. But turning right is not always optimally efficient: it may, for instance, increase the time for the vehicle to reach its destination. This increase in predictability should therefore coincide with an increase in safety, but a decrease in utility.

How can we balance these objectives while training?

#### Predictability Loss

Imagine a simulated road environment. In this environment, a self-driving car is learning to drive with RL, with various rewards related to driving performance. This agent has some model learning to plan. A variety of other agents exist in this environment, surrounding the planning agent. Each of these agents glances at the path of planning agent from its own perspective, and makes a partial observation of its trajectory.

These surrounding agents have their own model under training. They are now trained to predict the full trajectory of the planning agent, given their partial observations. Their negative prediction error is then given to the planning agent as an additional reward. This should cause the planning agent to attempt to increase the predictability of its actions, when under observation by nearby agents, regardless of their perspectives. And it should attempt to do so while satisfying the other rewards of basic driving.

This simultaneous dual network training scheme has some resemblance to a GAN. But the prediction networks could technically also be pre-trained and remain fixed.

### LLMs

With LLMs becoming agents capable of taking actions and pursuing goals, we seek a similar safety/utility balance. The analog to driving trajectory for a LLM could be considered its chain of thought. While an LLM should be capable of generating shrewd subgoals on the path to achieving objectives, we would like these subgoals to be within predictable bounds.

#### Predictability Loss for LLMs

A similar training scheme can apply for LLMs. Here we can use weaker or smaller models (SLM) to represent the surrounding human observers.

Imagine you have some LLM under training with RL. You then have one, or many, SLMs take partial observations of the output chains of thoughts of the LLM. In the same way as before, these now SLMs are trained to complete the trajectory of the LLM. Their negative prediction error is then given as an auxiliary reward to the LLM. This additional reward should cause the LLM to produce outputs in a style which remains predictable or interpretable to weaker models. It should (hopefully) still not restrict though, the LLMs ultimate performance on its main objectives. At the very least, they should balance.

This training scheme is an example of weak agents providing oversight to potentially much stronger ones. Human generated data could also substitute in for the SLMs in some cases.

In a simpler version, the SLMs are pre-trained and remain fixed. Here one can simply take the SLM logprob(LLM output) as the additional reward. Note that to a small extent, causal masking produces partial observations when computing logprobs.

Checkout `llm_safety.py` for a small example with one fixed SLM. Thanks to this [gist](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) for the basis of the code.