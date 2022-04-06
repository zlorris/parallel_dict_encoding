#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "verification.hu"

/**
 * @brief Verifies that the manually decoded file "decoded_manual.txt" in the
 *  /output directory is the same as the original input file "input.txt" in the
 *  /input directory.
 *
 * @param aNum number of words in the input.
 */
void manual_verify(unsigned int aNum)
{
  std::string dec_word, ref_word;

  // open the manual decoded file
  std::ifstream decoded_file("./output/decoded_manual.txt");
  if (!decoded_file.is_open())
  {
    std::cerr << "ERROR: Unable to open decoded file for manual verification!" << std::endl;
  }

  // open the original input file
  std::ifstream input_file("./input/input.txt");
  if (!input_file.is_open())
  {
    std::cerr << "ERROR: Unable to open original input file for manual verification!" << std::endl;
  }

  // verify that the decoded file is the same as the input file
  for (unsigned int i = 0; i < aNum; ++i)
  {
    std::getline(decoded_file, dec_word);
    std::getline(input_file, ref_word);

    assert(dec_word == ref_word);
  }
}