import sys
import os
import logging.config

# Avoid toolz dependency so this module loads in minimal envs (e.g. Colab clone)
_logging_conf = os.path.join(os.path.dirname(__file__), 'logging.conf')
if os.path.isfile(_logging_conf):
    logging.config.fileConfig(_logging_conf)
else:
    logging.basicConfig(level=logging.INFO, format='%(levelname)-10s %(message)s')
# todo: when release, set to USER
# logger = logging.getLogger('debug')
logger = logging.getLogger('user')

# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         # import tqdm
#         super().__init__(level)
#         # self.tqdm = tqdm.tqdm(total=1, unit="log", leave=False)
#
#     def emit(self, record):
#         try:
#             # msg = self.format(record)
#             # self.tqdm.write(msg, file=sys.stderr)
#             # self.tqdm.update()
#             # tqdm.tqdm.write(msg, file=sys.stderr)
#             self.flush()
#         except (KeyboardInterrupt, SystemExit):
#             raise
#         except Exception:
#             self.handleError(record)

# logger.handlers
pass
# pipe(
#     sys.stderr,
#     logging.StreamHandler,
#     # logger.addHandler
#     logger.addHandler
# )
