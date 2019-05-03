python actor_critic.py \
  --env_name=MountainCar-v0 \
  --root_dir=~/tmp/icm/MountainCar \
  --num_parallel_environments=30 \
  --num_iterations=1000000 \
  --gin_param='train_eval.use_icm=1' \
  --gin_param='train_eval.train_interval=100' \
  --gin_param='train_eval.actor_fc_layers=(256,)' \
  --gin_param='train_eval.value_fc_layers=(256,)' \
  --gin_param='train_eval.encoding_fc_layers=(200,)' \
  --gin_param='ICMAlgorithm.hidden_size=200' \
  --gin_param='suite_gym.load.max_episode_steps=1000' \
  --gin_param='EncodingNetwork.activation_fn=@tf.nn.elu' \
  --gin_param='ActorDistributionNetwork.activation_fn=@tf.nn.elu' \
  --gin_param='ValueNetwork.activation_fn=@tf.nn.elu' \
  --gin_param='ActorCriticLoss.entropy_regularization=0.1' \
  --gin_param='ActorCriticAlgorithm.gradient_clipping=10.0' \
  --gin_param='train_eval.debug_summaries=1' \
  --gin_param='train_eval.summarize_grads_and_vars=1' \
  --alsologtostderr
