#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "decoding.hu"

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
 * @brief Manually decodes encoded data file "encoded_manual.txt" into
 *  the file "decoded_manual.txt" in the /output directory.
 *
 * @param aNum number of words in the input
 */
void manual_decode(unsigned int aNum)
{
  std::unordered_map<unsigned int, std::string> reverse_dict;
  std::string word;

  // open the input file
  std::ifstream input_file("./output/encoded_manual.txt");
  if (!input_file.is_open())
  {
    std::cerr << "ERROR: Unable to open the input file for manual decoding!" << std::endl;
    exit(1);
  }

  // open the output file
  std::ofstream output_file("./output/decoded_manual.txt");
  if (!output_file.is_open())
  {
    std::cerr << "ERROR: Unable to open output file for manual decoding!" << std::endl;
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