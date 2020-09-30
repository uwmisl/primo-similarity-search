import multiprocessing.pool as mp
import itertools
import ipywidgets
from IPython.display import display
import logging

class ProgressPool(mp.Pool):

    def map(self, func, iterable):

        try:
            nitems = len(iterable)
            progress = ipywidgets.IntProgress(
                value = 0,
                min = 0,
                max = nitems,
                step = 1,
                description = "0/%d" % nitems
            )
            nitems = str(nitems)

        except TypeError:
            progress = ipywidgets.IntProgress(
                value = 1,
                min = 0,
                max = 1,
                description = "0/?",
                bar_style = 'info'
            )
            nitems = '?'

        display(progress)

        counter = itertools.count(1)
        def update_progress(result):
            count = next(counter)
            progress.value = count
            progress.description = "%d/%s" % (count, nitems)

        promises = []
        for args in iterable:

            promises.append(
                self.apply_async(func, (args,), callback=update_progress)
            )

        results = [None for _ in range(len(promises))]
        for ix, promise in enumerate(promises):

            try:
                results[ix] = promise.get()

            except Exception as e:
                progress.bar_style = 'danger'
                logging.exception("input %d failed" % ix)
                results[ix] = e

        if progress.bar_style != 'danger':
            progress.bar_style = 'success'
        return results


