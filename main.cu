#include "encoding.hu"
#include "decoding.hu"
#include "io.hu"
#include "verification.hu"

#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
  char *h_input = nullptr, *d_input = nullptr;
  unsigned int *h_indices = nullptr, *d_indices = nullptr;

  unsigned int input_size = 0;
  unsigned int input_num = 0;

  if (argc != 2)
  {
    std::cerr << "ERROR: Usage - ./main [--parallel, --manual]" << std::endl;
    exit(1);
  }

  // read the input into host memory
  read_input(&h_input, &h_indices, &input_size, &input_num);

  if (!strcmp(argv[1], "--parallel"))
  {
    // copy input to device memory
    cudaMalloc((void **)&d_input, input_size * sizeof(char));
    cudaMemcpy(d_input, h_input, input_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_indices, (input_num + 1) * sizeof(unsigned int));
    cudaMemcpy(d_indices, h_indices, (input_num + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // perform parallel encoding/decoding and verification
    parallel_encode(d_input, d_indices, input_num);
  }
  else if (!strcmp(argv[1], "--manual"))
  {
    // perform manual encoding/decoding and verification
    manual_encode(h_input, h_indices, input_num);
    manual_decode(input_num);
    manual_verify(input_num);

    for (unsigned int i = 0; i < input_num; ++i)
    {
      for (unsigned int j = h_indices[i]; j < h_indices[i + 1]; ++j)
      {
        std::cout << h_input[j] << std::endl;
      }
    }
  }
  else
  {
    std::cerr << "ERROR: Usage - ./main [parallel, manual]" << std::endl;
    exit(1);
  }

  // deallocate memory
  free(h_input);
  free(h_indices);
  // cudaFree(d_input);
  // cudaFree(d_indices);

  return 0;
}