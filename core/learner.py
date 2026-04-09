import os
import time
from arguments import Args
from core.buffer import ReplayBuffer as replay_buffer
from core.registry import get_algorithm

env_params = Args.env_params
train_params = Args.train_params

batch_size = train_params.batch_size
evalue_interval = train_params.evalue_interval
device = train_params.device
initial_eps = train_params.initial_eps
final_eps   = train_params.final_eps
decay_steps = train_params.decay_steps


def store_buffer(buffer, data_queue):
    for _ in range(data_queue.qsize()):
        buffer.push(data_queue.get(block=True))


def learn(model_path, data_queue, evalue_queue, actor_queues):
    from core.logger import Logger
    logger = Logger(logger="learner")
    algo = get_algorithm(Args.algo_name, Args, env_params, device=device)
    buffer = replay_buffer(env_params, train_params, logger)
    Actor_loss, Critic_loss = 0, 0
    Grad_norm_critic, Grad_norm_actor = 0.0, 0.0
    n_agents = env_params.n_agents
    Entropy = [0.0] * n_agents
    savetime = 0

    for queue in actor_queues:
        init_params = algo.get_actor_state_dict()
        init_params['current_eps'] = initial_eps
        queue.put(init_params)

    while buffer.current_size < batch_size:
        store_buffer(buffer, data_queue)
        logger.info(f'wating for samples... buffer current size {buffer.current_size}')
        time.sleep(5)

    for step in range(1, train_params.learner_step):
        store_buffer(buffer, data_queue)
        transitions = buffer.sample(batch_size)
        result = algo.update(transitions, logger, step=step)
        actor_loss, critic_loss = result[0], result[1]
        extra = result[2] if len(result) > 2 else {}
        Actor_loss += actor_loss
        Critic_loss += critic_loss
        Grad_norm_critic += extra.get('grad_norm_critic', 0.0)
        Grad_norm_actor  += extra.get('grad_norm_actor',  0.0)
        for i, v in enumerate(extra.get('entropy', [])):
            if i < n_agents:
                Entropy[i] += v

        if step % train_params.evalue_interval == 0:
            logger.info(f'epoch: {step // evalue_interval}, cur step: {step}')

        if step % evalue_interval == 0:
            current_eps = max(final_eps, initial_eps - (initial_eps - final_eps) * step / decay_steps)
            Actor_loss       /= evalue_interval
            Critic_loss      /= evalue_interval
            Grad_norm_critic /= evalue_interval
            Grad_norm_actor  /= evalue_interval
            Entropy_avg = [e / evalue_interval for e in Entropy]
            logger.info(f'epoch: {step // evalue_interval}, cur step: {step}, eps: {current_eps:.4f}, actor loss:{Actor_loss:.4f}, critic loss:{Critic_loss:.4f}')

            model_params = algo.get_actor_state_dict()
            model_params['current_eps'] = current_eps
            for queue in actor_queues:
                queue.put(model_params)

            evalue_params = {
                'actor_dict':              model_params['actor_dict'],
                'step':                    step,
                'actor_loss':              Actor_loss,
                'critic_loss':             Critic_loss,
                'grad_norm_critic':        Grad_norm_critic,
                'grad_norm_actor':         Grad_norm_actor,
                'policy_entropy_explorer': Entropy_avg[0] if n_agents > 0 else 0.0,
                'policy_entropy_postman':  Entropy_avg[1] if n_agents > 1 else 0.0,
                'policy_entropy_surveyor': Entropy_avg[2] if n_agents > 2 else 0.0,
            }
            evalue_queue.put(evalue_params)

            os.makedirs(model_path, exist_ok=True)
            algo.save(model_path + '/' + str(train_params.seed) + '_' + str(savetime) + '_model.pt')
            savetime += 1
            Actor_loss, Critic_loss = 0, 0
            Grad_norm_critic, Grad_norm_actor = 0.0, 0.0
            Entropy = [0.0] * n_agents
