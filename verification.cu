#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "verification.hu"

/**
 * @brief Verifies that the decoded file is the same as the input file
 *
 * @param d_file decoded filename
 * @param i_file input filename
 * @param aNum number of words in the files
 */
void verify(const char *d_file, const char *i_file, unsigned int aNum)
{
  std::string dec_word, ref_word;

  // open the decoded file
  std::ifstream decoded_file(d_file);
  if (!decoded_file.is_open())
  {
    std::cerr << "ERROR: Unable to open decoded file for verification!" << std::endl;
  }

  // open the original input file
  std::ifstream input_file(i_file);
  if (!input_file.is_open())
  {
    std::cerr << "ERROR: Unable to open input file for verification!" << std::endl;
  }

  // verify that the decoded file is the same as the input file
  for (unsigned int i = 0; i < aNum; ++i)
  {
    std::getline(decoded_file, dec_word);
    std::getline(input_file, ref_word);

    assert(dec_word == ref_word);
  }
}