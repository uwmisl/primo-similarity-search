import numpy as np
import pandas as pd

import cupyck
from cupyck.session.session import Session

from ..tools import sequences as seqtools

class Simulator:
    """
    Wrapper for running context (e.g. GPU, remote-execution).
    """

    defaults = {
        # Reverse Primer.
        "RP": "GTCCTCAACAACCTCCTG",
        # First 6 nucleotides of the reverse primer.
        "toehold": "GTCCTC",

        # Target molar concentration.
        "t_conc": 1e-9,
        # Query molar concentration.
        "q_conc": 1e-9,
        # Final temperature of the annealing process.
        "temp": 21
    }

    def __init__(self, sess_or_client, **kwargs):

        for arg, val in self.defaults.items():
            setattr(self, arg, val)

        for arg, val in kwargs.items():
            setattr(self, arg, val)

        if isinstance(sess_or_client, cupyck.Client):
            self.client = sess_or_client
            self.session = None

        elif isinstance(sess_or_client, Session):
            self.client = None
            self.session = sess_or_client

        else:
            raise ValueError("must provide valid session or client")

    def simulate(self, feature_seq_pairs):
        """
        Takes a batch of pairs of feature sequences as a pandas dataframe,
        simulates the thermodynamic yield from their

        """

        if self.client is not None:
            return self.client(feature_seq_pairs)

        # These are all simulation parameters that will be passed to the concentration session.
        conc_jobs = feature_seq_pairs.apply(
            lambda pair:
                { "sequences": [
                      pair.target_features + self.RP,
                      seqtools.revcomp(pair.query_features + self.toehold)
                  ],
                  "x0": np.array([self.t_conc, self.q_conc]),
                  "temperature": self.temp,
                  # Each strand can hybridize with a maximum of one other strand.
                  "max_complex_size": 2
                },
            axis = 1,
            result_type = 'expand'
        )

        conc_results = self.session.concentrations(conc_jobs)


        # Calculates the yield (ratio of final concentration of the duplex to the initial concentration of the limiting reagent).
        # Duplex here refers to the hybridized target and query.
        duplex_yields = conc_results.apply(
            lambda result:
                result.concentrations[(1,2)] / result.x0.min(),
            axis = 1
        )
        duplex_yields.name = "duplex_yield"

        return conc_jobs.join(duplex_yields)

if __name__ == "__main__":
    # This allows you to use this simulator as either a client or a server.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("max_seqlen", type=int)
    parser.add_argument("port", type=int)
    parser.add_argument("--nblocks", type=int)
    parser.add_argument("--nthreads", type=int)

    parser.add_argument("--rp", type=str)
    parser.add_argument("--toehold", type=str)
    parser.add_argument("--t_conc", type=float)
    parser.add_argument("--q_conc", type=float)
    parser.add_argument("--temp", type=float)

    args = parser.parse_args()

    sess_args = {
        "max_seqlen": args.max_seqlen,
        "nblocks": args.nblocks
    }
    if args.nthreads is not None:
        sess_args['nthreads'] = args.nthreads

    sim_args = {}
    if args.rp is not None:
        sim_args['rp'] = args.rp
    if args.toehold is not None:
        sim_args['toehold'] = args.toehold
    if args.t_conc is not None:
        sim_args['t_conc'] = args.t_conc
    if args.q_conc is not None:
        sim_args['q_conc'] = args.q_conc
    if args.temp is not None:
        sim_args['temp'] = args.temp

    try:
        session = cupyck.GPUSession(**sess_args)
    except RuntimeError:
        print "GPU startup failed. falling back to multicore backend."
        session = cupyck.MulticoreSession()

    simulator = Simulator(session, **sim_args)
    class SimServer(cupyck.Server):
        def worker(self, jobs):
            return simulator.simulate(jobs)

    server = SimServer(args.port, session)
    server.listen(verbose=True)