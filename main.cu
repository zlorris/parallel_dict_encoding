#include "encoding.hu"
#include "decoding.hu"
#include "io.hu"
#include "verification.hu"

#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
  char *h_input = nullptr;
  unsigned int *h_indices = nullptr;

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
    // perform parallel encoding/decoding and verification
    parallel_encode(h_input, h_indices, input_size, input_num);
  }
  else if (!strcmp(argv[1], "--manual"))
  {
    // perform manual encoding/decoding and verification
    manual_encode(h_input, h_indices, input_num);
    manual_decode(input_num);
    manual_verify(input_num);
  }
  else
  {
    std::cerr << "ERROR: Usage - ./main [parallel, manual]" << std::endl;
    exit(1);
  }

  // deallocate memory
  free(h_input);
  free(h_indices);

  return 0;
}