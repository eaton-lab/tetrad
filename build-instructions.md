


version='0.9.7'
git tag -a $version -m 'tag version $version'
git push --tags 

<!-- {% set version = "0.9.7" %} -->
wget -O- https://github.com/eaton-lab/tetrad/archive/0.9.7.tar.gz | shasum -a 256
conda build . -c bioconda -c conda-forge
anaconda upload     /path/to/tetrad-0.9.7-py312_1.tar.bz2 -u eaton-lab
