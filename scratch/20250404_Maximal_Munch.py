#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

def load_sparse_pauli_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Step 1: Extract the Pauli strings inside SparsePauliOp([...])
    match = re.search(r"SparsePauliOp\(\[([^\]]+)\]", content)
    if not match:
        raise ValueError("Could not find Pauli strings in file.")

    raw_strings = match.group(1).split(',')
    pauli_strings = [s.strip().strip("'").replace(" ", "") for s in raw_strings if s.strip()]

    return pauli_strings


# In[2]:


def maximal_munch(pauli_str):
    chunks = []
    start = None

    for i, c in enumerate(pauli_str):
        if c in {'X', 'Y', 'Z'}:
            if start is None:
                start = i
        else:
            if start is not None:
                chunks.append((start, pauli_str[start:i]))
                start = None

    if start is not None:
        chunks.append((start, pauli_str[start:]))

    return chunks


# In[3]:


def munch_file(filename):
    pauli_strings = load_sparse_pauli_file(filename)
    all_chunks = [(pstr, maximal_munch(pstr)) for pstr in pauli_strings]
    return all_chunks


# In[4]:


for pauli, munches in munch_file("hamiltonian_pauli.txt"):
    print(f"{pauli} → {munches}")


# In[ ]:




