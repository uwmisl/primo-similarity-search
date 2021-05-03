import cupyck
import pandas as pd
import numpy as np
import argparse

# TODO: This module is redundant with simulator.py, we thank it for its service,
# and it is time to let it go (e.g. delete it).
defaults = {
    "t_conc": 1e-9,
    "q_conc": 1e-9,
    "temperature": 21
}

def calc_yields(session, jobs, use_defaults = True):

    if use_defaults:
        jobs['t_conc'] = defaults['t_conc']
        jobs['q_conc'] = defaults['q_conc']
        jobs['temperature'] = defaults['temperature']

    conc_jobs = jobs.apply(
        lambda job:
            { "sequences": [job.target, job.query],
              "x0": np.array([job.t_conc, job.q_conc]),
              "temperature": job.temperature,
              "max_complex_size": 2
            },
        axis = 1,
        result_type = 'expand'
    )

    conc_results = session.concentrations(conc_jobs)

    duplex_yields = conc_results.apply(
        lambda result:
            result.concentrations[(1,2)] / result.x0.min(),
        axis = 1
    )
    duplex_yields.name = "duplex_yield"

    return jobs.join(duplex_yields)


class YieldServer(cupyck.Server):

    def worker(self, jobs):
        return calc_yields(self.session, jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("max_seqlen", type=int)
    parser.add_argument("port", type=int)
    parser.add_argument("--nblocks", type=int)
    parser.add_argument("--nthreads", type=int)
    args = parser.parse_args()

    sess_args = dict(
        max_seqlen = args.max_seqlen,
        nblocks = args.nblocks
    )

    if args.nthreads:
        sess_args['nthreads'] = args.nthreads

    try:
        session = cupyck.GPUSession(**sess_args)
    except RuntimeError:
        print "GPU startup failed. falling back to multicore backend."
        session = cupyck.MulticoreSession()

    server = YieldServer(args.port, session)
    server.listen(verbose=True)
