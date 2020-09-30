import os
import tempfile
import multiprocessing

import numpy as np

def decode_read_dir(read_dir, barcoder, ip_sequence="AGCACTCAGTATTTGTCCG"):
        
    is_gz = [ read_file.endswith(".gz") for read_file in os.listdir(read_dir) ]
    
    if all(is_gz):
        cat_cmd = "zcat %s/*.gz" % read_dir
    else:
        if any(is_gz):
            raise RuntimeError("mix of gzipped and plain read files not allowed")
            
        cat_cmd = "cat %s/*" % read_dir
    
    with tempfile.NamedTemporaryFile() as temp:
        
        # extract barcodes
        barcode_length = barcoder.total_seqlen

        subprocess.call(
            (cat_cmd + "| egrep -o '[ATCGN]{%d}%s' | cut -b 1-%d > %s") % (
                barcode_length, barcode_sequence, barcode_length, temp.name
            ),
            shell = True
        )

        barcodes = [ barcode.strip() for barcode in temp ]
        
        
    # decode
    pool = multiprocessing.Pool()
    try:
        results = np.array(pool.map(barcoder.seq_to_num, barcodes))
    finally:
        pool.close()
        
    decoded = results[results != None].astype(int)
    
    counts = np.bincount(decoded, minlength=1600000)[:1600000]
        
    return counts
        