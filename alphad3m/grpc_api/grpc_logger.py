"""GRPC logger, utility to log all messages.
"""

import collections
import functools
import string


def log_message_out(logger, msg, name):
    logger.info("< %s", name)
    for line in str(msg).splitlines():
        logger.info("< | %s", line)
    logger.info("  ------------------")


def log_message_in(logger, msg, name):
    logger.info("> %s", name)
    for line in str(msg).splitlines():
        logger.info("> | %s", line)
    logger.info("  ------------------")


def log_stream_out(logger, gen, name):
    for msg in gen:
        log_message_out(logger, msg, name)
        yield msg


def log_stream_in(logger, gen, name):
    for msg in gen:
        log_message_in(logger, msg, name)
        yield msg


def _wrap(logger, func):
    name = func.__name__

    @functools.wraps(func)
    def wrapped(self, request, context):
        log_message_in(logger, request, name)
        ret = func(self, request, context)
        if isinstance(ret, collections.Iterable):
            return log_stream_out(logger, ret, name)
        else:
            log_message_out(logger, ret, name)
            return ret

    return wrapped


def log_service(logger):
    def wrapper(klass):
        base, = klass.__bases__
        for name in dir(base):
            if name[0] not in string.ascii_uppercase:
                continue
            if name not in klass.__dict__:
                continue
            setattr(klass, name, _wrap(logger, klass.__dict__[name]))
        return klass
    return wrapper


class LoggingStub(object):
    def __init__(self, stub, logger):
        self._stub = stub
        self._logger = logger

    def __getattr__(self, item):
        return self._wrap(getattr(self._stub, item), item)

    def _wrap(self, method, name):
        def wrapper(out_msg):
            log_message_out(self._logger, out_msg, name)

            in_msg = method(out_msg)

            if isinstance(in_msg, collections.Iterable):
                return log_stream_in(self._logger, in_msg, name)
            else:
                log_message_in(self._logger, in_msg, name)
                return in_msg

        return wrapper
