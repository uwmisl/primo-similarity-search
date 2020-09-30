import numpy as np
import unireedsolomon as rs
from unireedsolomon.rs import GF2int
import struct

def has_hp(seq):
    return any([
        "".join([b,b]) in seq
        for b in list("ATCG")
    ])

def ends_ok(seq):
    return (
        (seq[0] == "A" or seq[0] == "C")
        and 
        (seq[-1] == "G" or seq[-1] == "T")
    )

def enumerate_seqs(n):
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
    def __init__(self, n_data_symbols, n_check_symbols, bits_per_symbol, bases_per_symbol, seed=42):
        
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
