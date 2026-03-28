import os
import time
import torch.multiprocessing as mp

from arguments import Args
from core.logger import Logger
from core.actor import actor_worker
from core.learner import learn
from core.evaluator import evaluate_worker


class Runner:
    def __init__(self, args, env_params, train_params, origin_obstacle_states):
        self.args = args
        self.env_params = env_params
        self.train_params = train_params
        self.origin_obstacle_states = origin_obstacle_states
        self.logger = Logger(logger="runner")

        self.model_path = os.path.join(train_params.save_dir, train_params.env_name)
        os.makedirs(self.model_path, exist_ok=True)

        self.ctx = mp.get_context("spawn")
        self.data_queue = self.ctx.Queue()
        self.evalue_queue = self.ctx.Queue()
        self.actor_queues = [self.ctx.Queue() for _ in range(train_params.actor_num)]

        self._processes = []

    def _build_processes(self):
        for i in range(self.train_params.actor_num):
            p = self.ctx.Process(
                target=actor_worker,
                args=(self.data_queue, self.actor_queues[i], i,
                      self.logger, self.origin_obstacle_states),
                daemon=True,
            )
            self._processes.append(p)

        self._processes.append(self.ctx.Process(
            target=learn,
            args=(self.model_path, self.data_queue, self.evalue_queue,
                  self.actor_queues, self.logger),
            daemon=True,
        ))

        self._processes.append(self.ctx.Process(
            target=evaluate_worker,
            args=(self.train_params, self.env_params, self.model_path,
                  self.train_params.evalue_time, self.evalue_queue,
                  self.logger, self.origin_obstacle_states),
            daemon=True,
        ))

    def _shutdown(self):
        self.logger.info("Shutting down all processes...")
        for p in self._processes:
            if p.is_alive():
                p.terminate()
        for p in self._processes:
            p.join(timeout=5)
        for q in [self.data_queue, self.evalue_queue] + self.actor_queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass
        self.logger.info("All processes terminated.")

    def run(self):
        self._build_processes()
        # start actors first, then learner, then evaluator
        actor_count = self.train_params.actor_num
        try:
            for i, p in enumerate(self._processes[:actor_count]):
                p.start()
                self.logger.info(f"Actor {i} started (pid={p.pid})")
                time.sleep(1)

            learner_proc = self._processes[actor_count]
            learner_proc.start()
            self.logger.info(f"Learner started (pid={learner_proc.pid})")
            time.sleep(2)

            eval_proc = self._processes[actor_count + 1]
            eval_proc.start()
            self.logger.info(f"Evaluator started (pid={eval_proc.pid})")

            learner_proc.join()
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received.")
        finally:
            self._shutdown()
