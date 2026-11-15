# -*- coding: utf-8 -*-
"""The step runner: one step at a time, on a thread of its own.

Every button that starts a step ends up here, whichever tab it is on - the
steps of the pipeline, the environment sources, the survey analyses. The
runner makes the worker and its thread, connects what the worker says to
whoever shows it, and sees to it that the thread ends as a thread has to:
when the step is done, when it is aborted, and when the plugin is unloaded
while it runs.

It is an object of its own - ``dock.steps`` - and knows nothing of the
form: it is handed the name of the step and the settings to run it with.
What a running step looks like in the panel (the buttons, the progress bar,
the log) is ``run_panel``.
"""

from qgis.PyQt.QtCore import QThread

from ..bambi_processing import ProcessingWorker
from ..gui_utils import let_thread_finish


class StepRunner:
    """Runs one step at a time.

    ``worker`` is the worker of the running step, ``None`` while no step
    runs; ``thread`` is the thread of the step that was started last.
    """

    def __init__(self):
        self.worker = None
        self.thread = None

    def is_running(self) -> bool:
        """True from :meth:`start` until the step is told to have
        :meth:`ended`."""
        return self.worker is not None

    def start(self, processor, step: str, config: dict, *, finished, error,
              progress, log, starting=None) -> None:
        """Run *step* of *processor* with *config* on a thread of its own.

        :param finished: called with ``(step, success)`` when the step is
            over - also after it failed or was cancelled
        :param error: called with ``(step, message)`` when it failed
        :param progress: called with the percentage that is done
        :param log: called with every line the step writes
        :param starting: called once everything is connected, right before
            the thread starts: the place to show that a step runs

        Whoever receives ``finished`` or ``error`` calls :meth:`ended`.
        """
        if self.is_running():
            raise RuntimeError(
                f"'{step}' cannot start: another step is still running")
        # This may be called from the end of the step before - from its
        # "finished", which arrives before its thread was told to quit. A
        # thread that is let go of while it runs takes QGIS down with it:
        # that one is kept until it has ended by itself.
        let_thread_finish(self.thread)

        self.thread = QThread()
        self.worker = ProcessingWorker(processor, step, config)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(finished)
        self.worker.error.connect(error)
        self.worker.progress.connect(progress)
        self.worker.log.connect(log)

        # Cleanup connections
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        if starting is not None:
            starting()
        self.thread.start()

    def ended(self) -> None:
        """The step has said that it is over: none is running any more.
        Its thread ends by the connections made in :meth:`start`."""
        self.worker = None

    def cancel(self) -> bool:
        """Ask the running step to stop. It does so at its next check and
        then reports ``finished`` like any other.

        :return: False when no step is running
        """
        if self.worker is None:
            return False
        self.worker.cancel()
        return True

    def shut_down(self) -> None:
        """Before the panel is removed (the plugin is unloaded or reloaded):
        cancel the running step and let its thread end by itself.

        The panel used to be deleted with the step still running, and its
        thread object with it - which Qt answers by aborting the process:
        unloading the plugin during a run took QGIS down. The step stops at
        its next check; until then its thread is kept alive outside the
        panel (``gui_utils.let_thread_finish``).
        """
        worker, thread = self.worker, self.thread
        self.worker = None
        self.thread = None
        if worker is not None:
            try:
                worker.cancel()
                for signal in (worker.finished, worker.error,
                               worker.progress, worker.log):
                    try:
                        signal.disconnect()
                    except (TypeError, RuntimeError):  # nosec B110
                        pass
                # ... but the thread still has to be told when it is done
                worker.finished.connect(thread.quit)
            except RuntimeError:  # nosec B110 - already deleted by Qt
                pass
        let_thread_finish(thread, worker, wait_ms=1000)
