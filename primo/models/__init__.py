_default_sequences = {
    # Forward Primer
    "FP": "GCCGACCAGTTTCCATAG",
    # Internal Primer
    "IP": "AGCACTCAGTATTTGTCCG",
    # Reverse Primer
    "RP": "GTCCTCAACAACCTCCTG",
    # First 6 bases of Reverse Primer(must change accordingly if changing Reverse Primer)
    "toehold": "GTCCTC",
    # (IDT-company specific notation) Stick a 5-Prime Biotin onto the tail
    # This is used during magnetic bead extraction.
    "biotin": "/5Biosg/UUUUUUUTT"
}

from encoder import Encoder
from predictor import Predictor
from encoder_trainer import EncoderTrainer
from simulator import Simulator