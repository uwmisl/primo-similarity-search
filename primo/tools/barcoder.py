import numpy as np
import unireedsolomon as rs
from unireedsolomon.rs import GF2int
import struct

def has_hp(seq):
    """Whether a DNA sequence contains a homopolymer (i.e. two or more nucleotides).

    Parameters
    ----------
    seq : str
        DNA sequence as a string.

    Returns
    -------
    boolean
        True if the sequence contains any homopolymers, False otherwise.
    """
    return any([
        "".join([b,b]) in seq
        for b in list("ATCG")
    ])

def ends_ok(seq):
    """Ensures that the sequence starts with an "A" or "C", and ends
    with "G" or "T". TODO: Document why this is necessary.

    Parameters
    ----------
    seq : str
        DNA sequence to check.

    Returns
    -------
    bool
        Whether the sequence starts and ends match expected values.
    """
    return (
        (seq[0] == "A" or seq[0] == "C")
        and
        (seq[-1] == "G" or seq[-1] == "T")
    )

def enumerate_seqs(n):
    """Enumerates all possible DNA sequences of length 'n'.

    Parameters
    ----------
    n : int
        Length of the DNA sequence.

    Returns
    -------
    List[str]
        List of all possible DNA sequences of length.
    """
    if n == 0:
        return [""]

    seqs = []
    for base in list("ATCG"):
        seqs += [
            "".join([base] + list(seq))
            for seq in enumerate_seqs(n-1)
        ]
    return seqs

def base10_to_baseK(num, K):
    """Converts a base-10 number to base-K.


    Parameters
    ----------
    num : int
        Base-10 number to convert to base K.
    K : int
        New base to use. Must be greater than 0.

    Returns
    -------
    List[int]
        A list of baseK digits that compose the original number.
    """
    if num == 0:
        return [0]

    rem = num % K
    result = []

    while num >= 1:
        result.append(rem)
        num /= K
        rem = num % K

    return result

def baseK_to_base10(num, K):
    result = 0
    for ix, n in enumerate(num):
        result += n * (K ** ix)
    return result

class Barcoder(object):
    """Converts numbers to DNA sequences with some error-correcting capabilities, and back again.
    """
    def __init__(self, n_data_symbols, n_check_symbols, bits_per_symbol, bases_per_symbol, seed=42):
        """[summary]

        Parameters
        ----------
        n_data_symbols : int
            Length of code word in symbols before any error-correcting.
        n_check_symbols : int
            Number of additional symbols used for error-correction.
        bits_per_symbol : int
            Number of bits per symbol.
        bases_per_symbol : int
            Number of nucleotides per symbol.
        seed : int, optional
            Random number seed, by default 42
        """
        self.n_data_symbols   = n_data_symbols
        self.n_check_symbols  = n_check_symbols
        self.bits_per_symbol  = bits_per_symbol
        self.bases_per_symbol = bases_per_symbol

        self.total_symbols    = (self.n_data_symbols + self.n_check_symbols)
        self.max_bits         = self.bits_per_symbol * self.n_data_symbols
        self.max_data         = 2**self.max_bits - 1
        self.total_seqlen     = self.total_symbols * self.bases_per_symbol

        self.ecc = rs.RSCoder(
            n=self.total_symbols,
            k=self.n_data_symbols,
            c_exp=self.bits_per_symbol,
            prim=rs.find_prime_polynomials(c_exp=self.bits_per_symbol)[2]
        )

        self.codebook = np.array([
            seq for seq in enumerate_seqs(self.bases_per_symbol)
            if not has_hp(seq) and ends_ok(seq)
        ])[:2**self.bits_per_symbol]

        assert len(self.codebook) == 2**self.bits_per_symbol

        # Spreads out the barcodes in the sequence space so that there are no implied
        # relations between adjacent numbers (e.g. the difference between 0 and 1 should be
        # arbitrary, but if we don't randomize then the sequences will be too similar).
        #
        # Note: This was developed out of an abundance of caution, preventing the introduction
        # of systematic biases in DNA synthesis or sequencing.
        np.random.seed(seed)
        self.barcodes   = np.random.permutation(self.max_data)
        self.unbarcodes = np.argsort(self.barcodes)


    def indices_to_encoder_vals(self, indices):
        if self.bits_per_symbol <= 8:
            vals = struct.pack("B" * len(indices), *indices)
        else:
            vals = map(GF2int, indices)
        return vals

    def encoder_vals_to_indices(self, encoder_vals):
        indices = []
        for val in encoder_vals:
            if type(val) == str:
                indices += struct.unpack("B", val)
            else:
                indices.append(int(val))

        return indices

    def num_to_seq(self, num):

        message_indices = base10_to_baseK(num, 2**self.bits_per_symbol)
        message_indices += [0]*(self.n_data_symbols-len(message_indices))

        message = self.indices_to_encoder_vals(message_indices)
        code = self.ecc.encode(message)
        code_indices = self.encoder_vals_to_indices(code)

        subseqs = self.codebook[code_indices]

        return "".join(subseqs)

    def seq_to_num(self, seq):

        subseqs = [
            "".join(seq)
            for seq in np.array_split(list(seq), self.total_symbols)
        ]

        try:

            code_indices = [
                np.where(self.codebook == subseq)[0][0]
                for subseq in subseqs
            ]

            if any([index > 2**self.bits_per_symbol for index in code_indices]):
                raise IndexError

            code = self.indices_to_encoder_vals(code_indices)
            message, ecc_val = self.ecc.decode(code, nostrip=True)
            message_indices = self.encoder_vals_to_indices(message)
            num = baseK_to_base10(message_indices, 2**self.bits_per_symbol)

            check = "OK" if self.ecc.check(message + ecc_val) else "Error Detected"

            return [check, num]

        except IndexError:
            return ["Invalid Codeword", None]
            raise
        except Exception:
            return ["ECC Failed", None]

    def num_to_barcode_seq(self, n):
        barcode = self.barcodes[n]
        seq = self.num_to_seq(barcode)
        return seq

    def barcode_seq_to_num(self, seq):
        check, num = self.seq_to_num(seq)
        if check == "OK":
            return self.unbarcodes[num]
        else:
            return None
