# LikelihoodsClusterAbundance

**Author:** Constantin Payerne
**Contact:** constantin.payerne@gmail.com

This repository for studying cluster abundance likelihoods

- The directory `\modules\...` containes the python modules for studying cluster abundance likeloods.

- The directory `\notebooks\Testing_likelihood_accuracy_with_PINOCCHIO\` contains the binned cluster abundance likelihood analysis that is presented in [Payerne et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.6223P/abstract). `\modules\pinocchio_analysis_testing_accuracy\` contains the job files. You can find a presentaion of this work 'Testing the likelihood accuracy for cluster abundance cosmology' at the [GDR on Cosmology](https://indico.ijclab.in2p3.fr/event/8881/contributions/28596/). 

- The directory `\modules\Unbinned_likelihood_with_SSC\` is dedicated to include Super-Sample Covariance in the unbinned likelihood for cluster count cosmology. This analysis is detailles in [Payerne et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240110024P/abstract) and was also presented at the [Journées Rubin LSST France](https://indico.in2p3.fr/event/30995/contributions/131829/).

## Requirements
This repository uses the following dependencies:
- [NumPy](https://www.numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [CCL](https://ccl.readthedocs.io/en/latest/)
- [PySSC](https://github.com/fabienlacasa/PySSC)