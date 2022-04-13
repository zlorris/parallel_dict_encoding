#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "encoding.hu"
#include "hash_table.hu"

#define HASH_ENTRIES 1024

/**
 * @brief Parallel encoding device kernel
 *
 * @param aInput flattened input character array on the device
 * @param aIndices array of indices in flattened array for each word on the device
 * @param aNum number of the words in the input
 * @param table pointer to device hash table
 * @param locks pointer to device hash table locks
 * @param results array of encoded indices on the device
 */
__global__ void parallel_encode_kernel(char *aInput, unsigned int *aIndices, unsigned int aNum,
                                       Table *table, Lock *locks, unsigned int *results)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  while (tid < aNum)
  {
    for (int i = 0; i < 32; i++)
    {
      if ((tid % 32) == i)
      {
        // make the key and value
        unsigned int length = aIndices[tid + 1] - aIndices[tid];
        void *key = aInput + aIndices[tid];
        void *val = aIndices + tid;

        // add the key and value to the table
        size_t hashValue = hash(table, key);
        locks[hashValue].lock();
        void *result = add_to_table(hashValue, key, length, val, table, locks, tid);
        locks[hashValue].unlock();

        // encoded index of word
        results[tid] = (unsigned int *)result - aIndices;
      }
    }

    tid += stride;
  }
}

/**
 * @brief Parallel encodes input data to file "encoded_parallel.txt" in the
 *  /output directory
 *
 * @param aInput flattened input character array on the host
 * @param aIndices array of indices in flattened array for each word on the host
 * @param aNum number of words in the input
 */
void parallel_encode(char *aInput, unsigned int *aIndices, unsigned int aSize, unsigned int aNum)
{
  char *d_input;
  unsigned int *d_indices, *h_results, *d_results;
  Table h_table;
  Table *d_table;
  Lock h_locks[HASH_ENTRIES];
  Lock *d_locks;

  // copy input to device memory
  cudaMalloc((void **)&d_input, aSize * sizeof(char));
  cudaMemcpy(d_input, aInput, aSize * sizeof(char), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_indices, (aNum + 1) * sizeof(unsigned int));
  cudaMemcpy(d_indices, aIndices, (aNum + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

  // allocate results arrays
  h_results = (unsigned int *)malloc(aNum * sizeof(unsigned int));
  cudaMalloc((void **)&d_results, aNum * sizeof(unsigned int));

  // initialize device table
  initialize_table(h_table, HASH_ENTRIES, aNum, false);
  cudaMalloc((void **)&d_table, sizeof(Table));
  cudaMemcpy(d_table, &h_table, sizeof(Table), cudaMemcpyHostToDevice);

  // initialize device locks
  cudaMalloc((void **)&d_locks, HASH_ENTRIES * sizeof(Lock));
  cudaMemcpy(d_locks, h_locks, HASH_ENTRIES * sizeof(Lock), cudaMemcpyHostToDevice);

  // perform parallel encoding/decoding and verification
  parallel_encode_kernel<<<4, 32>>>(d_input, d_indices, aNum, d_table, d_locks, d_results);

  // synchronize the host and device
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
  {
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    return;
  }

  // copy results back to host
  cudaMemcpy(h_results, d_results, aNum * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // open the output file
  std::ofstream output_file("./output/encoded_parallel.txt");
  if (!output_file.is_open())
  {
    std::cerr << "ERROR: Unable to open the output file for parallel encoding!" << std::endl;
    exit(1);
  }

  // write results to the output file
  for (unsigned int i = 0; i < aNum; ++i)
  {
    if (h_results[i] == i)
    {
      std::string word(aInput + aIndices[i], aInput + aIndices[i + 1]);
      output_file << word << std::endl;
    }
    else
    {
      output_file << h_results[i] << std::endl;
    }
  }

  output_file.close();

  // deallocate memory
  free_table(&h_table);
  cudaFree(d_input);
  cudaFree(d_indices);
  cudaFree(d_locks);
  cudaFree(d_table);
  free(h_results);
}

/**
 * @brief Manually encodes input data to file "encoded_manual.txt" in the
 *  /output directory.
 *sd array for each word
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