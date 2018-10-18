#converts wtb files to txt movelists
import numpy as np

def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='little')

def load_wtb_file(filename):
    dataset = open(filename, "rb").read()
    output = open(filename[:len(filename)-3]+"txt","w")

    print(dataset.shape)
    header = dataset[:16]
    numberOfRecords = bytes_to_int(header[4:7])
    body = dataset[16:]

    for x in range(numberOfRecords):
        #do stuff in chunks of 68-ish

    


    print(bytes_to_int(header[4:7]))
    print(header[1])

load_wtb_file("WTH_2004.wtb")
