import random
import yaml

"""
Script for generating input word list. Configuration parameters can be
edited in config.yaml:

file:
    inputFilepath: file path for list of input words
    outputFilepath: file path for output words
    metadataFilepath: file path for metadata of output words

parameters:
    bankSize: number of unique words to choose from the input for word bank
    outputSize: number of words to select from the word bank

"""

if __name__ == "__main__":
    # load the configuration file
    config = yaml.safe_load(open('config.yaml'))

    # parse all words from the input file
    words = []
    with open(config['file']['inputFilepath'], 'r') as words_file:
        for word in words_file:
            words.append(word)

    # form a word bank
    words_bank = random.sample(words, config['parameters']['bankSize'])

    # choose words from the bank
    words_chosen = []
    chosen_size = 0
    for i in range(config['parameters']['outputSize']):
        choice = random.choice(words_bank)
        words_chosen.append(choice)
        chosen_size += len(choice)

    # write output file
    with open(config['file']['outputFilepath'], 'w') as out_file:
        for word in words_chosen:
            out_file.write(word)

    # write output metadata file
    with open(config['file']['metadataFilepath'], 'w') as metadata_file:
        metadata_file.write(str(chosen_size) + '\n')
        metadata_file.write(str(config['parameters']['outputSize']) + '\n')
