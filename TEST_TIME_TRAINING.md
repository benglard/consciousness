# Test Time Training

You could potentially handle test time distribution shift if you could retrain your network at inference time. Prediction is a groundtruth that is always available, at training and inference time. So imagine a network with a shared trunk, a perception head, and a prediction head. Whenever you forward pass the network (training or inference) you first take a sequence of past frames and take a few gradient steps to optimize the shared weights and the prediction head. Then you take the last frame, run it through the shared weights and the perception head. At training time you can then compute a perception loss. This is a bi-level optimization at training time: outer loop optimizes perception, inner loop optimizes prediction. Essentially you are trying to train a base network that with a small number of optimization steps can reconfigure itself to perceive optimally in new domains.

If the additional head was control instead of perception it is basically adaptive mpc, except your "model" is an entire neural net.

Perhaps this is how the brain works 🤷‍♂️😃? It certainly alings with the idea of the brain as a prediction machine. And the belief that we do much of our training in-context with short-term memory, and then consolidate these learnings into long-term memory during sleep.

#### Extensions

The fast weights optimized at inference time do not have to be every weight of the base network, it could be selected layers or even new LoRA type parameters.