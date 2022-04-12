#include <cstring>

#include "hash_table.hu"
#include "./utility/error.hu"

__device__ __host__ size_t hash(Table *table, void *key)
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

__device__ void initialize_table(Table *table, size_t entries, size_t elements, bool reverse)
{
  table->num_entries = entries;
  table->num_elements = elements;
  table->reverse = reverse;

  table->entries = (Entry **)malloc(entries * sizeof(Entry *));
  memset(table->entries, 0, entries * sizeof(Entry *));
  table->pool = (Entry *)malloc(elements * sizeof(Entry));
  table->lock = (Lock *)malloc(elements * sizeof(Lock));
}

__host__ void copy_table_to_host(const Table *table, Table *hostTable)
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

__device__ __host__ void free_table(Table *table)
{
  free(table->pool);
  free(table->entries);
}

__device__ void add_to_table(void *key, size_t key_len, void *value,
                             Table *table, int tid, void **result)
{
  if (tid < table->num_elements)
  {
    size_t hashValue = hash(table, key);

    // check to see if the key already has an entry
    Entry *cur_entry = table->entries[hashValue];
    while (cur_entry != 0)
    {
      if (keys_equal(cur_entry->key, cur_entry->key_len, key, key_len))
      {
        break;
      }
    }

    if (cur_entry != 0)
    {
      // if an entry exists
      *result = cur_entry->value;
    }
    else
    {
      printf("table->pool: %lx\r\n", table->pool);
      // if no entry exists, make a new one
      Entry *location = &(table->pool[tid]);
      location->key = key;
      location->key_len = key_len;
      location->value = value;
      *result = value;
      table->lock[hashValue].lock();
      location->next = table->entries[hashValue];
      table->entries[hashValue] = location;
      table->lock[hashValue].unlock();
    }
  }
}

__device__ bool keys_equal(void *k1, size_t k1_len, void *k2, size_t k2_len)
{
  if (k1_len == 0 || k2_len == 0)
  {
    printf("no length\r\n");
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
