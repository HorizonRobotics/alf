Create INTERFACES.md and then write down interfaces of the following:
- Algorithm
- RLAlgorithm
- OnPolicyAlgorithm
- OffPolicyAlgorithm
- replay buffer (not sure how the class is called)

i.e. for each of those classes I need a list of (public) methods (except for those already in the parent class) for each saying the types (and shapes or whatever other invariants there are) of inputs and types/shapes/etc of their outputs, together with a brief 1-2 sentence  description of what it does

for example, if some methods does rollouts, I would be interested in what format does it accept states (are they batched, for instance), how it accepts agents (can it run different agents for differents states in batch, e.g.), what it returns, etc

because the library is *very* flexible, sometimes you might not be able to infer those invariants just from types (e.g. method my accept 
something like StateInfo, which is like "whatever"), but then you must look at the implementation to see what kinds of inputs it actually can work with (and save the relevant excerpts from the implementations to your report as well, e.g. showing that rollout method only runs the same agent on all states, or smth) -- it might even require multiple hops across files, be prepared.

aim to make report both concise (because there will be a lot to cover, need to save reading time) but informative (for the purpose of 
e.g. refactoring those interfaces, or introducing new abstractions into this library) 


file got too verbose. let's shorten INTERFACES.md, to preserve only crucial info for high-level thinking aobut concurrent RL abstraction,
  e.g.:\
  - no need to explain predict_step of Algorithm, if it just calls rollout_step\
  - no need to mention some technical args like `config` or `debug_summaries`, if they don't affect the relevant logic or constrain the
  architecture\