#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "encoding.hu"
#include "hash_table.hu"

#define HASH_ENTRIES 1024

__global__ void parallel_encode_kernel(char *aInput, unsigned int *aIndices,
                                       unsigned int aNum, Table &table, Lock *lock)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < aNum)
  {
    // make the key, value, and result
    unsigned int length = aIndices[tid + 1] - aIndices[tid];
    void *key = aInput + aIndices[tid];
    void *val = &tid;
    unsigned int index;
    void *result = &index;

    add_to_table(key, length, val, table, lock, tid, &result);

    printf("%u\r\n", *(unsigned int *)result);
  }

  __syncthreads();
}

/**
 * @brief Parallel encodes input data to file "encoded_parallel.txt" in the
 *  /output directory
 *
 * @param aInput flattened input character array on the device
 * @param aIndices array of indices in flattened array for each word on the device
 * @param aNum number of words in the input
 */
void parallel_encode(char *aInput, unsigned int *aIndices, unsigned int aNum)
{
  Table h_table, d_table;

  // initialize device hash table
  initialize_table(d_table, HASH_ENTRIES, aNum, false);

  // create a set of locks for the device hash table
  Lock lock[HASH_ENTRIES];
  Lock *d_lock;
  cudaMalloc((void **)&d_lock, HASH_ENTRIES * sizeof(Lock));
  cudaMemcpy(d_lock, lock, HASH_ENTRIES * sizeof(Lock), cudaMemcpyHostToDevice);

  // perform parallel encoding/decoding and verification
  parallel_encode_kernel<<<ceil(aNum / 512.0), 512>>>(aInput, aIndices, aNum, d_table, d_lock);

  // copy_table_to_host(d_table, h_table);

  // free tables
  // free_table(d_table);
}

/**
 * @brief Manually encodes input data to file "encoded_manual.txt" in the
 *  /output directory.
 *
 * @param aInput flattened input character array
 * @param aIndices array of indices in flattened array for each word
 * @param aNum number of words in the input
 */
void manual_encode(char *aInput, unsigned int *aIndices, unsigned int aNum)
{
  std::unordered_map<std::string, unsigned int> dict;

  // open the output file
  std::ofstream output_file("./output/encoded_manual.txt");
  if (!output_file.is_open())
  {
    std::cerr << "ERROR: Unable to open the output file for manual encoding!" << std::endl;
    exit(1);
  }

  // encode the input to the output file
  for (unsigned int i = 0; i < aNum; ++i)
  {
    std::string word(aInput + aIndices[i], aInput + aIndices[i + 1]);

    auto result = dict.insert(std::make_pair(word, i));

    if (result.second)
    {
      output_file << word << std::endl;
    }
    else
    {
      output_file << result.first->second << std::endl;
    }
  }

  output_file.close();
}