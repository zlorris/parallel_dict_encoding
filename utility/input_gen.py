import random
import yaml

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

    # choose words from the bank and write to output file
    words_chosen = [random.choice(words_bank)
                    for i in range(config['parameters']['outputSize'])]
    with open(config['file']['outputFilepath'], 'w') as out_file:
        for word in words_chosen:
            out_file.write(word)
