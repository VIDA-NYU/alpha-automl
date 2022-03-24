"""Version of multiprocessing that doesn't use fork.

Necessary because of grpcio bug.
"""

import base64
import importlib
import logging
import multiprocessing.connection
import os
import pickle
import socket
from queue import Empty
import subprocess
import traceback
import sys


class Receiver(object):
    def __init__(self):
        self._listener = socket.socket(getattr(socket, 'AF_UNIX'))
        try:
            self._listener.setsockopt(socket.SOL_SOCKET,
                                      socket.SO_REUSEADDR, 1)
            address = multiprocessing.connection.arbitrary_address('AF_UNIX')
            self._listener.bind(address)
            self._listener.listen(1)
            self.address = self._listener.getsockname()
        except OSError:
            self._listener.close()
            raise

        self._sockets = [self._listener]

    def recv(self, timeout=None):
        done = False
        while not done:
            done = True
            for sock in multiprocessing.connection.wait(self._sockets,
                                                        timeout):
                if sock == self._listener:
                    s, addr = self._listener.accept()
                    conn = multiprocessing.connection.Connection(s.detach())
                    self._sockets.append(conn)
                    done = False
                else:
                    try:
                        msg = sock.recv()
                    except EOFError:
                        self._sockets.remove(sock)
                    else:
                        return msg
        raise Empty

    def send(self, msg):
        for sock in self._sockets:
            if sock != self._listener:
                sock.send(msg)

    def close(self):
        try:
            self._listener.close()
        finally:
            os.unlink(self.address)


def run_process(target, tag, msg_queue, **kwargs):
    """Call a Python function by name in a subprocess.

    :param target: Fully-qualified name of function to call.
    :param tag: Tag to add to logger to identify that process.
    :return: A `subprocess.Popen` object.
    """
    assert isinstance(msg_queue, Receiver)
    data = msg_queue.address, kwargs
    proc = subprocess.Popen(
        [
            sys.executable,
            '-c',
            'from alphad3m.multiprocessing import _invoke; _invoke(%r, %r)' % (
                tag, target
            ),
            base64.b64encode(pickle.dumps(data)),
        ],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    return proc


def _invoke(tag, target):
    """Invoked in the subprocess to setup logging and start the function.

    Arguments are read from ``sys.argv``.
    """
    data = pickle.loads(base64.b64decode(sys.argv[1]))
    address, kwargs = data

    tag = '{}-{}'.format(tag, os.getpid())

    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:{}:%(name)s:%(message)s".format(tag),
        stream=sys.stdout)

    msg_queue = multiprocessing.connection.Client(address)

    module, function = target.rsplit('.', 1)
    module = importlib.import_module(module)
    function = getattr(module, function)

    try:
        function(msg_queue=msg_queue, **kwargs)
    except Exception:
        logging.exception("Uncaught exception in subprocess %s", tag)
        error = traceback.format_exc()
        sys.stderr.write(error)
        sys.exit(1)
