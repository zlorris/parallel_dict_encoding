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

  unsigned int input_size = 0, input_num = 0;

  if (argc != 2)
  {
    std::cerr << "ERROR: Usage - ./main [--parallel, --serial]" << std::endl;
    exit(1);
  }

  // read the input into host memory
  read_input("./input/input.txt", "./input/input_metadata.txt", &h_input, &h_indices, &input_size, &input_num);

  if (!strcmp(argv[1], "--parallel"))
  {
    // perform parallel encoding/decoding and verification
    parallel_encode(h_input, h_indices, input_size, input_num);
    parallel_decode();
    verify("./output/decoded_parallel.txt", "./input/input.txt", input_num);
  }
  else if (!strcmp(argv[1], "--serial"))
  {
    // perform serial encoding/decoding and verification
    serial_encode(h_input, h_indices, input_num);
    serial_decode(input_num);
    verify("./output/decoded_serial.txt", "./input/input.txt", input_num);
  }
  else
  {
    std::cerr << "ERROR: Usage - ./main [parallel, serial]" << std::endl;
    exit(1);
  }

  // deallocate memory
  free(h_input);
  free(h_indices);

  return 0;
}