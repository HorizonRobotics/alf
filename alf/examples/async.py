import random
import time

import tensorflow as tf
import tensorflow.nest as nest
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

import threading

random.seed(0)
tf.random.set_seed(0)

# configuration
_LEARN_QUEUE_CAP_ = 4
_NUM_ACT_QUEUES_ = 2
_ACT_QUEUE_CAP_ = 8
_NUM_ENVS_ = 16

_CENTRALIZED_ACTOR_ = True
_UNROLL_LENGTH_ = 2 # env unroll length
_EPS_ = 0.05        # exploration epsilon


class DQN(Model):
    def __init__(self, widths, out_dim):
        super(DQN, self).__init__()
        fcs = [layers.Dense(w, activation='relu') for w in widths]
        q_value = layers.Dense(out_dim)
        fcs.append(q_value)
        self.out_dim = out_dim
        self.model = Sequential(fcs)

    def call(self, x):
        return self.model(x)

    def predict(self, x):
        return tf.argmax(self(x), axis=1, output_type=tf.int32)


class Statistics(object):
    def __init__(self):
        self.iterations = 0 #tf.constant(0)
        self.success = 0 #tf.constant(0)
        self.total_episodes = 0 #tf.constant(0)

    def record(self, rewards):
        def episodes():
            return int(tf.reduce_sum(tf.abs(rewards)))

        def success():
            return int(tf.reduce_sum(tf.cast(tf.equal(rewards, 1.), tf.int32)))

        self.iterations += 1
        self.total_episodes += episodes()
        self.success += success()

    def rate(self):
        if int(self.total_episodes) == 0:
            return 0
        return int(self.success) / int(self.total_episodes)


class Array(object):
    """
    A simple 1D array game:

    In each episode the observation is a onehot vector of `self.len`. The agent starts
    from the center of the array and decides to move left or right at each step. The
    nonzero value of the vector indicates the agent's current position.

    Before an episode begins, the vector might be multiplied by -1 with a probablity of 0.5.

    If the vector contains a 1, then the agent should move towards left end;
    otherwise if the vector contains a -1, it should move towards right end.

    A success/failure gives the agent a reward of 1/-1. No time penalty.

    A random/fixed agent will have an expected success rate of 0.5.
    """
    def __init__(self, length):
        self.len = length
        self.reset()

    def observation(self):
        # shape: 1 x self.len
        return tf.cast(tf.one_hot([self.pos], self.len), tf.float32) * self.flag

    def reset(self):
        self.pos = self.len // 2
        self.flag = random.choice([-1, 1])
        # ob, reward, end (-1: unrolling start, 0: alive, 1: episode end)
        return self.observation(), 0., 0

    def step(self, action):
        self.pos += (-1 if action == 0 else 1)

        end = 0
        reward = 0
        if self.pos == 0:
            end = 1
            reward = self.flag
        if self.pos == self.len - 1:
            end = 1
            reward = -self.flag

        ob = self.observation()
        if end: # if end, return the initial ob
            ob, *_ = self.reset()
        return ob, reward, end


# actor thread
def run_actor(coord, dqn, tfq, actor_id, eps=_EPS_):
    def get_actions(obs):
        pred_actions = dqn.predict(obs)
        random_actions = tf.random.uniform(shape=pred_actions.shape,
                                           minval=0,
                                           maxval=dqn.out_dim,
                                           dtype=tf.int32)
        random_real = tf.random.uniform(shape=random_actions.shape,
                                        minval=0,
                                        maxval=1,
                                        dtype=tf.float32)
        actions = tf.where(random_real < eps,
                           random_actions,
                           pred_actions)
        return actions

    with coord.stop_on_exception():
        while not coord.should_stop():
            obs, env_ids = tfq.act_queues[actor_id].dequeue_many(_ACT_QUEUE_CAP_)
            obs = tf.concat(tf.unstack(obs, axis=0), axis=0)
            actions = get_actions(obs)
            for env_id, act in tf.stack([env_ids, actions], axis=1):
                tfq.act_return_queues[env_id].enqueue(act)

        # Whoever stops first, cancel all pending requests
        # (including enqueues and dequeues),
        # so that no thread hangs before calling coord.should_stop()
        tfq.close_all()


def unroll_env(env, unroll_length, dqn, centralized_actor,
               tfq=None, env_id=None, eps=_EPS_):
    """
    Unroll the env for a cerntain length
    If `dqn` is not None, then this function will directly forward the model
    otherwise, it will let the actor to forward the model
    """
    def step(input, _):
        ob = input[0]

        if centralized_actor:
            act_queue = random.choice(tfq.act_queues)
            act_queue.enqueue([ob, env_id])
            action = int(tfq.act_return_queues[env_id].dequeue())
        else:
            if random.uniform(0, 1) < eps:
                action = random.choice([0, 1])
            else:
                action = int(dqn.predict(ob))

        # make sure to convert from tf scaler to python scalar!
        ob, reward, end = env.step(action)
        return ob, reward, end, action

    input = (env.observation(), 0., -1, 0)
    output = tf.scan(step, tf.range(unroll_length), input)
    # append the init input to the beginning of output
    output = nest.map_structure(lambda i, o: tf.concat([[i], o], axis=0),
                                input, output)
    # squeeze the extra dim of observation
    output = [tf.squeeze(output[0], axis=1)] + list(output[1:])
    return output


# env thread
#
# NOTE: because we potentially have *many* env threads, and
# Python threads share a CPU, so make sure the thread is lightweight
# and IO bound!
# If the env simulation is computation-heavy, consider moving the env
# simulator to an external process
def run_env(coord, dqn, env_f, tfq, total_unroll_steps, unroll_length, env_id):

    with coord.stop_on_exception():
        env = env_f()
        steps = 0
        while not coord.should_stop() and steps < total_unroll_steps:
            unrolled = unroll_env(
                env, unroll_length, dqn, _CENTRALIZED_ACTOR_,
                tfq, env_id)
            tfq.learn_queue.enqueue(unrolled)
            steps += unroll_length

        # For simplicity, we terminate all if one actor finishes
        coord.request_stop()
        # Whoever stops first, cancel all pending requests
        # (including enqueues and dequeues),
        # so that no thread hangs before calling coord.should_stop()
        tfq.close_all()


# learning thread
def run_learner(coord, dqn, tfq):
    with coord.stop_on_exception():
        optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        stat = Statistics()

        @tf.function
        def _train_batch(exps):
            next_exps = nest.map_structure(lambda t: t[1:], exps)
            exps = nest.map_structure(lambda t: t[:-1], exps)
            obs, _, _, _ = exps
            next_obs, rewards, ends, actions = next_exps

            starts = tf.cast(tf.equal(ends, -1), tf.float32)
            ends = tf.cast(tf.equal(ends, 1), tf.float32)

            next_values = dqn(next_obs)
            next_values = tf.reduce_max(next_values, axis=1) * 0.99
            target_values = rewards + next_values * (1 - ends)

            with tf.GradientTape() as tape:
                values = dqn(obs)
                actions = tf.stack([tf.range(actions.shape[0]), actions], axis=1)
                actual_values = tf.gather_nd(values, actions)
                # If exps come from multiple actors, we don't want to compute
                # TD across actors, thus multiplying (1 - starts)
                loss = tf.losses.mean_squared_error(
                    target_values * (1 - starts), actual_values * (1 - starts))

            var = dqn.trainable_variables
            grads = tape.gradient(loss, var)
            optimizer.apply_gradients(tuple(zip(grads, var)))

        def _statistics(exps, interval):
            rewards = exps[1]
            stat.record(rewards)
            if int(stat.iterations % interval) == 0:
                tf.print("current success rate: ", stat.rate())
                # terminate when the rate hits a threshold
                if stat.rate() > 0.99 and stat.total_episodes >= 100:
                    coord.request_stop()

        while not coord.should_stop():
            # dequeue will stack elements along axis=0
            exps = nest.map_structure(lambda t: tf.concat(tf.unstack(t, axis=0), axis=0),
                                      tfq.learn_queue.dequeue_many(_LEARN_QUEUE_CAP_))
            _statistics(exps, interval=50)
            _train_batch(exps)

        # Whoever stops first, cancel all pending requests
        # (including enqueues and dequeues),
        # so that no thread hangs before calling coord.should_stop()
        tfq.close_all()


class TFQueues(object):
    def __init__(self, num_envs, num_act_queues, dqn, env_f):
        """
        Create three kinds of queues:
        1. one learner queue
           - stores batches of training trajectories
             all agent threads should enqueue unrolled trajectories into it
        2. one actor queue
           - stores batches of observations to act upon
             all agent threads should enqueue current observations into it to
             get predicted actions
        3. `num_envs` action-returning queues
           - each env holds one such queue for receiving the returned action
             predicted the by actor

        args:
           num_envs - number of environments running in parallel
           dqn      - the model
           env_f    - creates a new environment when called
        """

        # unroll once to get dtypes and shapes for the queues
        traj = unroll_env(env_f(), _UNROLL_LENGTH_, dqn, False)

        self.learn_queue = tf.queue.FIFOQueue(
            capacity=_LEARN_QUEUE_CAP_,
            dtypes=nest.map_structure(lambda t: t.dtype, traj),
            shapes=nest.map_structure(lambda t: t.shape, traj)
        )

        assert num_envs >= num_act_queues * _ACT_QUEUE_CAP_, "Not enough environments!"
        self.act_queues = [
            tf.queue.FIFOQueue(
            capacity=_ACT_QUEUE_CAP_,
            dtypes=[traj[0].dtype, tf.int32],  # [observation, env_id]
            shapes=[[1, traj[0].shape[1]], []] # single observation and id scalar
            )
            for i in range(num_act_queues)]
        self.act_return_queues = [
            tf.queue.FIFOQueue(
                capacity=1,
                dtypes=traj[3].dtype, # action only
                shapes=[])            # single action scalar
            for i in range(num_envs)]

    def close_all(self):
        self.learn_queue.close(cancel_pending_enqueues=True)
        for aq in self.act_queues:
            aq.close(cancel_pending_enqueues=True)
        for arq in self.act_return_queues:
            arq.close(cancel_pending_enqueues=True)


def start_all_threads_and_wait(threads, coord):
    for t in threads:
        t.start()

    t0 = time.time()
    try:
        coord.join(threads)
    except KeyboardInterrupt as e:
        coord.request_stop()

    print("Time elapsed: {}".format(time.time() - t0))


if __name__ == "__main__":

    dqn = DQN(widths=[32], out_dim=2)
    env_f = lambda : Array(21)

    tf_queues = TFQueues(_NUM_ENVS_, _NUM_ACT_QUEUES_, dqn, env_f)

    # for coordinating among threads
    coord = tf.train.Coordinator()

    actor_threads = [threading.Thread(name="actor_thread",
                                      target=run_actor,
                                      args=(coord,
                                            dqn,
                                            tf_queues,
                                            i))
                     for i in range(_NUM_ACT_QUEUES_)]
    learner_thread = threading.Thread(name="learner_thread",
                                      target=run_learner,
                                      args=(coord,
                                            dqn,
                                            tf_queues))
    total_unroll_steps = 1000000
    env_threads = [threading.Thread(name="env_thread{}".format(i),
                                    target=run_env,
                                    args=(coord,
                                          dqn,
                                          env_f,
                                          tf_queues,
                                          total_unroll_steps,
                                          _UNROLL_LENGTH_,
                                          i))
                   for i in range(_NUM_ENVS_)]

    threads = actor_threads + [learner_thread] + env_threads

    start_all_threads_and_wait(threads, coord)
