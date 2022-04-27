# parallel_dict_encoding

## Project
***Title:*** Parallelized Dictionary Encoding </br>
***Class:*** ECSE 4740: Applied Parallel Computing for Engineers - Spring 2022</br>
***Contributors***: Zach Orris and Cohen Davis

## Instructions
### Input Generation
To generate input files, adjust the parameters in the _utility/config.yaml_ file:
* `file`
  * `inputFilepath`: file path for list of input words
  * `outputFilepath`: file path for output words
  * `metadataFilepath`: file path for metadata of output words
* `parameters`
  * `bankSize`: number of unique words to choose from the input to form a word bank
  * `outputSize`: number of words to select from the word bank
  
Run the _utility/input_gen.py_ script with these parameters.

### Program
To compile, use `make` with the provided *Makefile*.
To run, use one of the following:
* `./main --parallel` - Perform parallel encoding/decoding (on the GPU) and verification
* `./main --parallel_cpu` - Perform parallel encoding/decoding (on the CPU) and verification
* `./main --serial` - Perform serial encoding/decoding and verification

Note: While the parallel_cpu encoding has been configured to utilize the ideal number of threads for the files tested, one can change this number of threads by simply editing the number associated with the "#define CPU_THREADS" statement in main.cu.
