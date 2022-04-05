#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief Reads input file "input.txt" in the /input directory
 *  into @param aInput. The file "input_metadata.txt" contains
 *  information on the number of characters and number of words,
 *  which are stored in @param aSize and @param aNum, respectively.
 *
 * @param aInput pointer to flattened input character array
 * @param aIndices pointer to array of indices in flattened array for each word
 * @param aSize pointer to number of characters in the input
 * @param aNum pointer to number of words in the input
 *
 **/
void read_input(char **aInput, unsigned int **aIndices, unsigned int *aSize, unsigned int *aNum)
{
  std::string line;
  unsigned int char_cnt = 0, word_cnt = 0;

  // open the metadata file
  std::ifstream metadata_file("./input/input_metadata.txt");
  if (!metadata_file.is_open())
  {
    std::cerr << "ERROR: Unable to open input metadata file!" << std::endl;
    exit(1);
  }

  // read the metadata file
  std::getline(metadata_file, line);
  *aSize = std::stoul(line);
  std::getline(metadata_file, line);
  *aNum = std::stoul(line);
  metadata_file.close();

  // allocate input arrays
  *aInput = (char *)calloc(*aSize, sizeof(char));
  *aIndices = (unsigned int *)calloc(*aNum, sizeof(unsigned int));

  if (aInput == nullptr || aIndices == nullptr)
  {
    std::cerr << "ERROR: Unable to allocate input arrays!" << std::endl;
    exit(1);
  }

  // open the input file
  std::ifstream input_file("./input/input.txt");
  if (!input_file.is_open())
  {
    std::cerr << "ERROR: Unable to open input file!" << std::endl;
    exit(1);
  }

  // read the input file
  while (!input_file.eof())
  {
    std::getline(input_file, line);

    (*aIndices)[word_cnt] = char_cnt;

    for (unsigned int i = 0; i < line.size(); ++i)
    {
      (*aInput)[char_cnt] = line[i];
      char_cnt++;
    }

    word_cnt++;
  }
  input_file.close();
}
