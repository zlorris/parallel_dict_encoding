#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "decoding.hu"
#include "hash_table.hu"
#include "io.hu"

namespace
{
  /**
   * @brief Checks to see if a string is numeric.
   *
   * @param s input string
   */
  bool isNumber(const std::string &s)
  {
    return std::all_of(s.begin(), s.end(),
                       [](char c)
                       { return std::isdigit(c) != 0; });
  }
}

/**
 * @brief Parallel decoding kernel - builds hash table
 *
 * @param aInput flattened input character array on the device
 * @param aIndices array of indices in flattened array for each word on the device
 * @param aNum number of the words in the input
 * @param table pointer to device hash table
 * @param locks pointer to device hash table locks
 * @param results array of encoded indices on the device
 */
__global__ void parallel_decode_build_kernel(char *aInput, unsigned int *aIndices, unsigned int aNum,
                                             Table *table, Lock *locks, unsigned int *results)
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  while (tid < aNum)
  {
    for (int i = 0; i < 32; i++)
    {
      // letter words must be put into the hash table first
      if ((tid % 32) == i && aInput[aIndices[tid]] - '0' > 9)
      {
        // make the key and value
        void *key = aIndices + tid;
        void *val = aIndices + tid;

        // add the key and value to the table
        size_t hashValue = hash(table, key);
        locks[hashValue].lock();
        void *result = add_to_table(hashValue, key, 0, val, table, locks, tid);
        locks[hashValue].unlock();

        // encoded index of word
        results[tid] = (unsigned int *)result - aIndices;
      }
    }

    tid += stride;
  }
}

/**
 * @brief Parallel decoding kernel - looks up words in hash table
 *
 * @param aInput flattened input character array on the device
 * @param aIndices array of indices in flattened array for each word on the device
 * @param aNum number of the words in the input
 * @param table pointer to device hash table
 * @param locks pointer to device hash table locks
 * @param results array of encoded indices on the device
 */
__global__ void parallel_decode_lookup_kernel(char *aInput, unsigned int *aIndices, unsigned int aNum,
                                              Table *table, Lock *locks, unsigned int *results)
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  while (tid < aNum)
  {
    for (int i = 0; i < 32; i++)
    {
      // index words should lookup their real word in the hash table second
      if ((tid % 32) == i && aInput[aIndices[tid]] - '0' <= 9)
      {
        // make the key and value
        unsigned int index = 0;
        for (int j = 0; j < (aIndices[tid + 1] - aIndices[tid]); ++j)
        {
          index += pow(10, aIndices[tid + 1] - aIndices[tid] - j - 1) * (aInput[aIndices[tid] + j] - '0');
        }
        void *key = aIndices + index;
        void *val = aIndices + tid;

        // add the key and value to the table
        size_t hashValue = hash(table, key);
        locks[hashValue].lock();
        void *result = add_to_table(hashValue, key, 0, val, table, locks, tid);
        locks[hashValue].unlock();

        // decoded index of word
        results[tid] = (unsigned int *)result - aIndices;
      }
    }

    tid += stride;
  }
}
void parallel_decode()
{
  char *h_input, *d_input;
  unsigned int *h_indices, *d_indices, *h_results, *d_results;
  unsigned int input_size, input_num;
  Table h_table;
  Table *d_table;
  Lock h_locks[HASH_ENTRIES];
  Lock *d_locks;

  // read the encoded input
  read_input("./output/encoded_parallel.txt", "./output/encoded_parallel_metadata.txt",
             &h_input, &h_indices, &input_size, &input_num);

  // copy input to device memory
  cudaMalloc((void **)&d_input, input_size * sizeof(char));
  cudaMemcpy(d_input, h_input, input_size * sizeof(char), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_indices, (input_num + 1) * sizeof(unsigned int));
  cudaMemcpy(d_indices, h_indices, (input_num + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

  // allocate results arrays
  h_results = (unsigned int *)malloc(input_num * sizeof(unsigned int));
  cudaMalloc((void **)&d_results, input_num * sizeof(unsigned int));

  // initialize device table
  initialize_table(h_table, HASH_ENTRIES, input_num, true);
  cudaMalloc((void **)&d_table, sizeof(Table));
  cudaMemcpy(d_table, &h_table, sizeof(Table), cudaMemcpyHostToDevice);

  // initialize device locks
  cudaMalloc((void **)&d_locks, HASH_ENTRIES * sizeof(Lock));
  cudaMemcpy(d_locks, h_locks, HASH_ENTRIES * sizeof(Lock), cudaMemcpyHostToDevice);

  // launch parallel decoding build kernel
  parallel_decode_build_kernel<<<ceil(input_num / 64.0), 64>>>(d_input, d_indices, input_num, d_table, d_locks, d_results);

  // launch parallel decoding lookup kernel
  parallel_decode_lookup_kernel<<<ceil(input_num / 64.0), 64>>>(d_input, d_indices, input_num, d_table, d_locks, d_results);

  // synchronize the host and device
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
  {
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    return;
  }

  // copy results back to host
  cudaMemcpy(h_results, d_results, input_num * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // open the output file
  std::ofstream output_file("./output/decoded_parallel.txt");
  if (!output_file.is_open())
  {
    std::cerr << "ERROR: Unable to open the output file for parallel decoding!" << std::endl;
    exit(1);
  }

  // write the results to the output file
  for (int i = 0; i < input_num; ++i)
  {
    char *start = h_input + h_indices[h_results[i]];
    unsigned int length = h_indices[h_results[i] + 1] - h_indices[h_results[i]];

    std::string word(start, start + length);
    output_file << word << std::endl;
  }

  output_file.close();

  // deallocate memory
  free_table(&h_table);
  cudaFree(d_input);
  cudaFree(d_indices);
  cudaFree(d_locks);
  cudaFree(d_table);
  cudaFree(d_results);
  free(h_input);
  free(h_indices);
  free(h_results);
}

/**
 * @brief Serially decodes encoded data file "encoded_serial.txt" into
 *  the file "decoded_serial.txt" in the /output directory.
 *
 * @param aNum number of words in the input
 */
void serial_decode(unsigned int aNum)
{
  std::unordered_map<unsigned int, std::string> reverse_dict;
  std::string word;

  // open the input file
  std::ifstream input_file("./output/encoded_serial.txt");
  if (!input_file.is_open())
  {
    std::cerr << "ERROR: Unable to open the input file for serial decoding!" << std::endl;
    exit(1);
  }

  // open the output file
  std::ofstream output_file("./output/decoded_serial.txt");
  if (!output_file.is_open())
  {
    std::cerr << "ERROR: Unable to open output file for serial decoding!" << std::endl;
  }

  // decode the input to the output file
  for (unsigned int i = 0; i < aNum; ++i)
  {
    std::getline(input_file, word);

    if (isNumber(word))
    {
      word = reverse_dict.find(std::stoul(word))->second;
    }
    else
    {
      reverse_dict.insert(std::make_pair(i, word));
    }

    output_file << word << std::endl;
  }

  input_file.close();
  output_file.close();
}
