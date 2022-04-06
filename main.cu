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

  read_input(&h_input, &h_indices, &input_size, &input_num);

  // Perform manual encoding/decoding and verification
  manual_encode(h_input, h_indices, input_num);
  manual_decode(input_num);
  manual_verify(input_num);

  return 0;
}