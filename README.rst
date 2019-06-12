

tetrad
------
Phylogenetic species tree inference using phylogenetic invariants and quartet joining

Description
-----------
Tetrad employs the inference method originally developed in SVDquartets for 
inferring quartet trees using phylogenetic invariants, and joining those 
quartet trees into a super tree to provide a topology that is statistically 
consistent under the multispecies coalescent model. 

Features
--------
Tetrad offers a number of advantages over SVDquartets:

1. Easy installation (conda).
2. Simple command line tool.
3. Optimizes SNP information for each quartet set (see [SNP-sampling](#snp-sampling)).
4. Bootstrap re-sampling samples both loci and SNPs (ideal for RAD data).
5. Fast: inference can be massively parallelized easily (e.g., MPI)

Installation
------------
.. code:: bash

	conda install tetrad -c eaton-lab -c conda-forge

Usage
-----
.. code:: bash

	# run tetrad on a snps data file from ipyrad
	tetrad -s data.snps.hfd5 

	# run on 80 cores distributed over 4 nodes on a cluster
	tetrad -s data.snps.hfd5 -c 80 --MPI


SNP sampling
------------
To reduce the effects of linked SNPs on the results tetrad should be 
used with linkage information turned on (it is by default), so that
only a single SNP is sampled from each locus. Tetrad samples a single
SNP from each locus that is segregating within each quartet set of 
of samples, and repeats this sampling on each iteration (e.g., bootstrap)
to maximize the amount of information used for each quartet resolution.

