#include "encoding.hu"
#include "decoding.hu"
#include "io.hu"
#include "verification.hu"

#include <iostream>
#include <string>
#include <chrono>
#define CPU_THREADS 2

int main(int argc, char *argv[])
{
  char *h_input = nullptr;
  unsigned int *h_indices = nullptr;

  unsigned int input_size = 0, input_num = 0;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;

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
    start_time = std::chrono::high_resolution_clock::now();
    parallel_encode(h_input, h_indices, input_size, input_num);
    end_time = std::chrono::high_resolution_clock::now();
    parallel_decode();
    verify("./output/decoded_parallel.txt", "./input/input.txt", input_num);
  }
  else if (!strcmp(argv[1], "--parallel_cpu")) 
  {
    // perform parallel encoding/decoding and verification on the CPU
    start_time = std::chrono::high_resolution_clock::now();
    parallel_cpu_encode(h_input, h_indices, input_size, input_num, CPU_THREADS);
    end_time = std::chrono::high_resolution_clock::now();
    parallel_cpu_decode(input_num, CPU_THREADS);
    verify("./output/decoded_parallel_cpu.txt", "./input/input.txt", input_num);
  }
  else if (!strcmp(argv[1], "--serial"))
  {
    // perform serial encoding/decoding and verification
    start_time = std::chrono::high_resolution_clock::now();
    serial_encode(h_input, h_indices, input_num);
    end_time = std::chrono::high_resolution_clock::now();
    serial_decode(input_num);
    verify("./output/decoded_serial.txt", "./input/input.txt", input_num);
  }
  else
  {
    start_time = std::chrono::high_resolution_clock::now();
    std::cerr << "ERROR: Usage - ./main [parallel, serial]" << std::endl;
    end_time = std::chrono::high_resolution_clock::now();
    exit(1);
  }

  //Calculate & print encoding processing time (wall clock time, not CPU time)
  auto process_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
  printf("Encoding took %.6f seconds\n", process_time.count() * 1e-9);

  // deallocate memory
  free(h_input);
  free(h_indices);

  return 0;
}