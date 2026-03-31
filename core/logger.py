import logging
import os
import os.path
import time
import csv
import colorama
from colorama import Fore, Style
import sys

colorama.init()

class Logger(object):
    def __init__(self, logger):
        self.start_time = time.time()
        self.logger = logging.getLogger(name=logger)
        self.logger.setLevel(logging.DEBUG) 
        rq = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        log_path = os.getcwd() + "/logs/"
        log_name = log_path + rq + ".log"
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(filename)s[line:%(lineno)d] - %(name)s - %(message)s")
            # 新增loss
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            os.makedirs(log_path, exist_ok=True)
            fh = logging.FileHandler(log_name, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _get_past_time(self):
        s_time = int(time.time() - self.start_time)
        day = s_time // (24 * 3600)
        s_time = s_time % (24 * 3600)
        hour = s_time // 3600
        s_time = s_time % 3600
        minutes = s_time // 60
        s_time = s_time % 60
        return f'day {day} - {hour}h:{minutes}m:{s_time}s'

    def debug(self, msg):
        self.logger.debug(Fore.WHITE + "DEBUG - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)
        
    def info(self, msg):
        self.logger.info(Fore.GREEN + "INFO - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)

    def warning(self, msg):
        self.logger.warning(Fore.RED + "WARNING - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)

    def error(self, msg):
        self.logger.error(Fore.RED + "ERROR - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)

    def critical(self, msg):
        self.logger.critical(Fore.RED + "CRITICAL - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)


# 评估指标字段顺序（同时作为 CSV 表头）
_EVAL_FIELDS = ['step', 'actor_loss', 'critic_loss',
                'success_rate', 'mean_coverage', 'mean_reward', 'mean_time',
                'mean_energy', 'mean_collision', 'mean_distance', 'fitness']


def log_eval_metrics(plot_path: str, metrics: dict):
    """
    将评估指标写入 TensorBoard 和 CSV。
    - TensorBoard: {plot_path}/tb/
    - CSV:         {plot_path}/eval_metrics.csv  (首次写入自动生成表头)
    """
    # --- TensorBoard ---
    try:
        import importlib
        SummaryWriter = importlib.import_module('torch.utils.tensorboard').SummaryWriter
        tb_dir = os.path.join(plot_path, 'tb')
        writer = SummaryWriter(log_dir=tb_dir)
        step = metrics.get('step', 0)
        for key in _EVAL_FIELDS:
            if key in metrics and key != 'step':
                writer.add_scalar(f'eval/{key}', metrics[key], global_step=step)
        writer.close()
    except Exception:
        pass  # TensorBoard 不可用时静默跳过

    # --- CSV ---
    os.makedirs(plot_path, exist_ok=True)
    csv_path = os.path.join(plot_path, 'eval_metrics.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_EVAL_FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


if __name__ == '__main__':
    log = Logger(logger="test")
    log.debug("debug")
    log.info("info")
    log.error("error")
    log.warning("warning")
    log.critical("asdasdasdqwfqf")