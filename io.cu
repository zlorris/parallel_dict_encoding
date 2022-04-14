#include <fstream>
#include <iostream>
#include <string>

#include "io.hu"

/**
 * @brief Reads input file @param i_file into @param aInput. The
 *  metadata file @param m_file contains information on the number of
 *  characters  and number of words,  which are stored in @param aSize
 *  and @param aNum, respectively.
 *
 * @param i_file input filename
 * @param m_file metadata filename
 * @param aInput pointer to flattened input character array
 * @param aIndices pointer to array of indices in flattened array for each word
 * @param aSize pointer to number of characters in the input
 * @param aNum pointer to number of words in the input
 *
 **/
void read_input(const char *i_file, const char *m_file, char **aInput,
                unsigned int **aIndices, unsigned int *aSize, unsigned int *aNum)
{
  std::string line;
  unsigned int char_cnt = 0, word_cnt = 0;

  // open the metadata file
  std::ifstream metadata_file(m_file);
  if (!metadata_file.is_open())
  {
    std::cerr << "ERROR: Unable to open metadata file!" << std::endl;
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
  *aIndices = (unsigned int *)calloc(*aNum + 1, sizeof(unsigned int));

  if (aInput == nullptr || aIndices == nullptr)
  {
    std::cerr << "ERROR: Unable to allocate input arrays!" << std::endl;
    exit(1);
  }

  // open the input file
  std::ifstream input_file(i_file);
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

  // add a final entry to indices for the last word
  (*aIndices)[word_cnt] = char_cnt;

  input_file.close();
}
