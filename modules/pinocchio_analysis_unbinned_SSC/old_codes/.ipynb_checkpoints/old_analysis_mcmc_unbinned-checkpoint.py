analysis = {}

#full
wheretosave = '/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/manuscript/'
#binned
redshift_bins = [0.2, 1.2]
logm_bins =  [14.2, 15.6]

analysis['1'] = {"name":"unbinnedSSC_high1", 
                 'likelihood': 'unbinnedSSC', 
                 'redshift_bins':[0.2, 1.2], 
                 'logm_bins':[14.2, 15.6], 
                 'where_to_save': wheretosave,
                 'cat_name': '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/converted_plc_mock0451.dat'}
analysis['2'] = {"name":"unbinnedSSC_high2", 
                 'likelihood': 'unbinnedSSC', 
                 'redshift_bins':[0.2, 1.2] , 
                 'logm_bins':[14.2, 15.6], 
                 'where_to_save': wheretosave,
                 'cat_name': '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/converted_plc_mock0269.dat'}
analysis['3'] = {"name":"unbinnedSSC_high3", 
                 'likelihood': 'unbinnedSSC', 
                 'redshift_bins':[0.2, 1.2] , 
                 'logm_bins':[14.2, 15.6], 
                 'where_to_save': wheretosave,
                 'cat_name':'/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/converted_plc_mock0077.dat'}


#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 16]]
#analysis['1'] = {"name":"full", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#small
#redshift_bins = [[0.2, 0.4]]
#logm_bins =     [[14.2, 14.4]]
#analysis['2'] = {"name":"small", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#lowredshift
#redshift_bins = [[0.2, 0.4]]
#logm_bins =     [[14.2, 16],]
#analysis['3'] = {"name":"low-redshift", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save':wheretosave}

#lowmass
#redshift_bins = [[0.2, 1]]
#logm_bins =     [[14.2, 14.4],]
#analysis['4'] = {"name":"low-mass",'redshift_bins':redshift_bins , 'logm_bins':logm_bins,  'where_to_save':wheretosave}

#highmass
#redshift_bins = [[0.2, 1]]
#logm_bins =     [[14.5, 16],]
#analysis['5'] = {"name":"high-mass", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save':wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 14.3]]
#analysis['1'] = {"name":"mass_bin_1", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 14.4]]
#analysis['2'] = {"name":"mass_bin_2", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 14.5]]
#analysis['3'] = {"name":"mass_bin_3", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 14.6]]
#analysis['4'] = {"name":"mass_bin_4", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 14.7]]
#analysis['5'] = {"name":"mass_bin_5", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 14.8]]
#analysis['6'] = {"name":"mass_bin_6", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 14.9]]
#analysis['7'] = {"name":"mass_bin_7", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 15]]
#analysis['8'] = {"name":"mass_bin_8", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}

#redshift_bins = [[0.2, 1]]
#logm_bins =  [[14.2, 16]]
#analysis['9'] = {"name":"mass_bin_9", 'redshift_bins':redshift_bins , 'logm_bins':logm_bins, 'where_to_save': wheretosave}







