#!/usr/bin/env python

"""ipyrad Cluster class for starting/stopping w/ ipyparallel.

This uses the new ipyparallel cluster API (v.>7.0) for starting and
stopping clusters using a context wrapper. Our custom superclass
of the ipyparallel.Cluster class suppresses some print statements
of that tool, and adds our own custom logging and exception handling.
"""

from typing import Optional, Type
import re
import os
import sys
import time
import signal
import traceback
from datetime import timedelta
from loguru import logger
from ipyparallel.cluster.cluster import Cluster as IPCluster
from ipyparallel import RemoteError, Client



class Cluster(IPCluster):
    """Custom superclass of ipyparallel cluster.

    This class is used to start an ipcluster with an optional set of
    additional kwargs, return a connected Client instance, and
    shutdown the ipcluster when the context manager closes.

    Compared to the ipyparallel parent class, this one suppresses
    print statements and instead uses a logger, and the context manager
    exit function has custom exception handling and formatting for
    ipyrad specifically.
    """
    # suppress INFO calls from ipyparallel built-in logging.
    log_level = 30

    def __init__(self, cores: int, stop_timeout: float = 2., **kwargs):

        # cores is an alias for .n, which is also stored for ipp parent.
        self.n = cores or self._get_num_cpus()
        self.stop_timeout = stop_timeout  # Timeout for stopping engines

        # init parent class with kwargs for ipcluster start (e.g., MPI)
        super().__init__(**kwargs)

        # hidden attributes
        self._client_start_time: Optional[float] = None
        self._context_client: Optional[Client] = None
        self._engine_pids: list[int] = []  # Store PIDs of running engines        

    def __enter__(self):
        """Starts a new cluster and connects a client."""
        logger.info("Establishing parallel ipcluster...")
        self.start_controller_sync()
        self.start_engines_sync()
        client = self._context_client = self.connect_client_sync()
        if self.n:
            # wait for engine registration
            client.wait_for_engines(
                self.n,
                interactive=False,
                block=True,
                timeout=self.engine_timeout,
            )
            logger.info(f"{len(client)} engines started")
        logger.debug(f"Engine PIDs stored: {self._engine_pids}")
        self._client_start_time = time.time()
        return client

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Ensures shutdown of ipcluster and handling of exceptions."""
        try:
            if self._context_client:
                # Abort future jobs and interrupt running jobs with SIGTERM signal
                self._context_client.abort(block=True)
                self._context_client.send_signal(signal.SIGTERM, block=True)
                self._context_client.close()
                self._context_client = None

            # stop.
            self.stop_engines_sync()
            self.stop_controller_sync()
            elapsed = int(time.time() - self._client_start_time)
            elapsed = str(timedelta(seconds=elapsed))
            logger.info(f"ipcluster stopped. Elapsed time: {elapsed}")
        except Exception as shutdown_error:
            logger.error(f"Error during shutdown: {shutdown_error}")
        finally:
            if exc_value:
                self._handle_exception(exc_type, exc_value)

    def _handle_exception(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
    ) -> None:
        """Handle exceptions raised within the context manager."""
        if exc_type is KeyboardInterrupt:
            logger.error("Keyboard interrupt by user. Cleaning up.")
        elif exc_type == RemoteError:
            trace = "\n".join(exc_value.render_traceback())
            if not self._supports_color():
                trace = self._strip_ansi(trace)
            logger.error(f"An error occurred on an ipengine:\n{trace}")
        else:
            trace = traceback.format_exc()
            logger.error(f"An error occurred:\n{trace}")

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI color codes from a string."""
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    @staticmethod
    def _supports_color() -> bool:
        """Determine if the terminal supports color output."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @staticmethod
    def _get_num_cpus() -> int:
        """Get the number of CPUs available on the system."""
        return os.cpu_count()


# def get_num_cpus():
#     """Return the effective number of CPUs in the system.

#     Returns an integer for either Unix or MacOSX (Darwin). This code
#     is mostly copied from a similar implementation in IPython.
#     If it can't find a sensible answer, it returns 1.
#     """
#     if platform.system() == "Linux":
#         ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
#     else:
#         proc = subprocess.run(
#             ['sysctl', '-n', 'hw.ncpu'], check=True, capture_output=True)
#         ncpus = proc.stdout.decode().strip()
#     try:
#         ncpus = max(1, int(ncpus))
#     except:
#         ncpus = 1
#     return ncpus


# def color_support():
#     """Check for color support in stderr as a notebook or terminal/tty."""
#     # check if we're in IPython/jupyter
#     tty1 = bool(IPython.get_ipython())
#     # check if we're in a terminal
#     tty2 = sys.stderr.isatty()
#     return tty1 or tty2


if __name__ == "__main__":

    with Cluster(cores=0) as c:
        time.sleep(1)
        raise KeyboardInterrupt("interrupted.")
        print(c)
