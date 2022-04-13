#include <cassert>
#include <cstring>

#include "hash_table.hu"
#include "./utility/error.hu"

/**
 * @brief Compute the hash value for an element's key
 *
 * @param table pointer to the device hash table
 * @param key pointer to the element key
 */
__device__ size_t hash(Table *table, void *key)
{
  if (table->reverse)
  {
    // type casting - keys are unsigned int
    unsigned int *int_key = (unsigned int *)key;

    return (*int_key) % table->num_entries;
  }
  else
  {
    // djb2 hash function by Dan Bernstein
    // type casting - keys are char*
    char *str_key = (char *)key;

    unsigned long hash = 5381;
    int c;

    while (c = (*str_key)++)
      hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash % table->num_entries;
  }
}

/**
 * @brief Initialize the hash table
 *
 * @param table pointer to the hash table
 * @param entries number of entries for the hash table
 * @param elements number of elements for the hash table
 * @param reverse encoding (false) or decoding (true)
 */
void initialize_table(Table &table, size_t entries, size_t elements, bool reverse)
{
  table.num_entries = entries;
  table.num_elements = elements;
  table.reverse = reverse;

  cudaMalloc((void **)&table.entries, entries * sizeof(Entry *));
  cudaMemset(table.entries, 0, entries * sizeof(Entry *));
  cudaMalloc((void **)&table.pool, elements * sizeof(Entry));
}

/**
 * @brief Copy device hash table to the host
 *
 * @param table pointer to device hash table
 * @param hostTable pointer to host hash table
 */
void copy_table_to_host(const Table *table, Table *hostTable)
{
  hostTable->num_entries = table->num_entries;
  hostTable->num_elements = table->num_elements;
  hostTable->reverse = table->reverse;
  hostTable->entries = (Entry **)malloc(table->num_entries * sizeof(Entry *));
  hostTable->pool = (Entry *)malloc(table->num_elements * sizeof(Entry));

  cudaMemcpy(hostTable->entries, table->entries,
             table->num_entries * sizeof(Entry *),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hostTable->pool, table->pool,
             table->num_elements * sizeof(Entry),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < table->num_entries; i++)
  {
    if (hostTable->entries[i] != NULL)
      hostTable->entries[i] =
          (Entry *)((size_t)hostTable->entries[i] -
                    (size_t)table->pool + (size_t)hostTable->pool);
  }
  for (int i = 0; i < table->num_elements; i++)
  {
    if (hostTable->pool[i].next != NULL)
      hostTable->pool[i].next =
          (Entry *)((size_t)hostTable->pool[i].next -
                    (size_t)table->pool + (size_t)hostTable->pool);
  }
}

/**
 * @brief Free table memory allocated on the device
 *
 * @param table pointer to the hash table object
 */
void free_table(Table *table)
{
  cudaFree(table->pool);
  cudaFree(table->entries);
}

/**
 * @brief Add a new element to the hash table
 *
 * @param hash_value hash value of the element key (0 - table->num_entries)
 * @param key void pointer to the element key
 * @param key_len length of the element key
 * @param value void pointer to the element value
 * @param table pointer to device hash table
 * @param locks pointer to device locks
 * @param tid thread ID
 *
 * @return value of the element's entry in the hash table
 */
__device__ void *add_to_table(size_t hash_value, void *key, size_t key_len, void *value,
                              Table *table, Lock *locks, int tid)
{
  // check to see if the key already has an entry
  Entry *cur_entry = table->entries[hash_value];
  while (cur_entry != 0)
  {
    if (keys_equal(cur_entry->key, cur_entry->key_len, key, key_len))
    {
      break;
    }
    cur_entry = cur_entry->next;
  }

  if (cur_entry != 0)
  {
    // if an entry exists
    return cur_entry->value;
  }
  else
  {
    // if no entry exists, make a new one
    Entry *location = &(table->pool[tid]);
    location->key = key;
    location->key_len = key_len;
    location->value = value;

    // update the table
    location->next = table->entries[hash_value];
    table->entries[hash_value] = location;

    return value;
  }
}

/**
 * @brief Check if the keys of two elements are equal
 *
 * @param k1 pointer to the first element key
 * @param k1_len length of the first element key
 * @param k2 pointer to the second element key
 * @param k2_len length of the second element key
 */
__device__ bool keys_equal(void *k1, size_t k1_len, void *k2, size_t k2_len)
{
  if (k1_len == 0 || k2_len == 0)
  {
    unsigned int *k1_int = (unsigned int *)k1;
    unsigned int *k2_int = (unsigned int *)k2;

    return *k1_int == *k2_int;
  }
  else
  {
    char *k1_str = (char *)k1;
    char *k2_str = (char *)k2;

    if (k1_len != k2_len)
    {
      return false;
    }

    for (size_t i = 0; i < k1_len; ++i)
    {
      if (k1_str[i] != k2_str[i])
      {
        return false;
      }
    }
    return true;
  }
}
