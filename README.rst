

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
3. Optimizes SNP information for each quartet set (missing data no problem).
4. Bootstrap re-sampling samples both loci and SNPs (ideal for RAD data).
5. Fast: inference can be massively parallelized easily (e.g., MPI)

Installation
------------
. code:: bash
	conda install tetrad -c eaton-lab -c conda-forge

Usage
-----
.. code:: bash

	# run tetrad on a snps data file from ipyrad
	tetrad -s data.snps.hfd5 

	# run on 80 cores distributed over 4 nodes on a cluster
	tetrad -s data.snps.hfd5 -c 80 --MPI

