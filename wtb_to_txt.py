#converts wtb files to txt movelists
import numpy as np

def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='little')

def load_wtb_file(filename):
    dataset = open(filename, "rb").read()
    output = open(filename[:len(filename)-3]+"txt","w")

    header = dataset[:16]
    numberOfRecords = bytes_to_int(header[4:7])
    body = dataset[16:]

    for x in range(numberOfRecords):
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
        start_index = x * 68
        end_index = start_index + 67
        record = body[start_index:end_index]
        for i in range(0,60,2):
            output.write(str(bytes_to_int(record[8+i:9+i])-11))
            output.write(' ')
        output.write('\n')


load_wtb_file("WTH_2004.wtb")
load_wtb_file("WTH_2005.wtb")