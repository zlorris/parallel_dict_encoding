#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "encoding.hu"

__global__ void parallel_encode(char *aInput, unsigned int *aIndices, unsigned int aNum)
{
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