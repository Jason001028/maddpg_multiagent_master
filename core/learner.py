import time
from arguments import Args
from core.buffer import ReplayBuffer as replay_buffer
from core.registry import get_algorithm

env_params = Args.env_params
train_params = Args.train_params

batch_size = train_params.batch_size
evalue_interval = train_params.evalue_interval
device = train_params.device


def store_buffer(buffer, data_queue):
    for _ in range(data_queue.qsize()):
        buffer.push(data_queue.get(block=True))


def learn(model_path, data_queue, evalue_queue, actor_queues, logger):
    algo = get_algorithm(Args.algo_name, Args, env_params, device=device)
    buffer = replay_buffer(env_params, train_params, logger)
    Actor_loss, Critic_loss = 0, 0
    savetime = 0

    for queue in actor_queues:
        queue.put(algo.get_actor_state_dict())

    while buffer.current_size < batch_size:
        store_buffer(buffer, data_queue)
        logger.info(f'wating for samples... buffer current size {buffer.current_size}')
        time.sleep(5)

    for step in range(1, train_params.learner_step):
        store_buffer(buffer, data_queue)
        transitions = buffer.sample(batch_size)
        actor_loss, critic_loss = algo.update(transitions, logger, step=step)
        Actor_loss += actor_loss
        Critic_loss += critic_loss

        if step % train_params.update_tar_interval == 0:
            logger.info(f'cur step: {step}')

        if step % evalue_interval == 0:
            Actor_loss /= evalue_interval
            Critic_loss /= evalue_interval
            logger.info(f'cur step: {step}, actor loss:{Actor_loss:.4f}, critic loss:{Critic_loss:.4f}')

            model_params = algo.get_actor_state_dict()
            for queue in actor_queues:
                queue.put(model_params)

            evalue_params = {
                'actor_dict': model_params['actor_dict'],
                'step': step,
                'actor_loss': Actor_loss,
                'critic_loss': Critic_loss,
            }
            evalue_queue.put(evalue_params)

            algo.save(model_path + '/' + str(train_params.seed) + '_' + str(savetime) + '_model.pt')
            savetime += 1
            Actor_loss, Critic_loss = 0, 0
