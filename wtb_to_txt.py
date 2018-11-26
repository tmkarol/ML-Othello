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

    print(numberOfRecords)

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
        if record[6] > 32:
            output.write(str(1))
        elif record[6] == 32:
            output.write(str(0))
        else:
            output.write(str(-1))
        output.write(' ')
        for i in range(0,60,2):
            move = str(bytes_to_int(record[8+i:9+i])-11).zfill(2)
            output.write(move) #move lists
            output.write(' ')
        output.write('\n')

        #output is the game score for black followed by the movelist

load_wtb_file("WTH_2004.wtb")
load_wtb_file("WTH_2005.wtb")