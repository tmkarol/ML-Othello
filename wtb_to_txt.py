#converts wtb files to txt movelists
import numpy as np

def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='little')

def load_wtb_file(filename):
    dataset = open(filename, "rb").read()
    output = open(filename[:len(filename)-3]+"txt","w")

    #print(dataset.shape)
    header = dataset[:16]
    numberOfRecords = bytes_to_int(header[4:7])
    body = dataset[16:]

    for x in range(numberOfRecords):
        #do stuff in chunks of 68-ish
        start_index = x * 68
        end_index = start_index + 67
        record = body[start_index:end_index]
        '''
        Each record (68 bytes) contains:

        Label Size Type

        Word 2 tournament label number
        Player number Black 2 Word
        Player number White 2 Word
        Number of black pawns (real score) 1 Byte
        Theoretical score 1 Byte
        List of shots 60 Byte []
        '''


    #print(bytes_to_int(header[4:7]))
    #print(header[1])

load_wtb_file("WTH_2004.wtb")
