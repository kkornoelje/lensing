import numpy as np, sys, os, scipy as sc, argparse
sys_path_folder='../python/'
sys.path.append(sys_path_folder)

import flatsky, tools, lensing, foregrounds, misc

from tqdm import tqdm

from pylab import *
cmap = cm.RdYlBu_r

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)

#==========================================================================================================================#
parser = argparse.ArgumentParser(description='')
parser.add_argument('-start', dest='start', action='store', help='start', type=int, default=0)
parser.add_argument('-end', dest='end', action='store', help='end', type=int, default=10)
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, required=True)
parser.add_argument('-clusters_or_randoms', dest='clusters_or_randoms', action='store', help='clusters_or_randoms', type=str, default='clusters')
parser.add_argument('-random_seed_for_sims', dest='random_seed_for_sims', action='store', help='random_seed_for_sims', type=int, default=679)

args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]

    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

if clusters_or_randoms == 'randoms':
    start, end = 0, 1

#==========================================================================================================================#
param_dict = misc.get_param_dict(paramfile)
for key, value in param_dict.items():
    globals()[key] = value


nx = int(boxsize_am/dx)
mapparams = [nx, nx, dx]
x1,x2 = -nx/2. * dx, nx/2. * dx
verbose = 0


if not pol:
    tqulen = 1
else:
    tqulen = 3
tqu_tit_arr = ['T', 'Q', 'U']

#get ra, dec or map-pixel grid
ra=np.linspace(x1,x2, nx) #arcmins
dec=np.linspace(x1,x2, nx) #arcmins
ra_grid, dec_grid=np.meshgrid(ra,dec)

el, cl = tools.get_cmb_cls(cls_file, pol = pol)

if clusters_or_randoms == 'clusters':
    print('\tgetting NFW convergence for lensing')
    #NFW lensing convergence
    ra_grid_deg, dec_grid_deg = ra_grid/60., dec_grid/60.

    M200c_list = np.tile(cluster_mass, total_clusters)
    redshift_list = np.tile(cluster_z, total_clusters)
    ra_list = dec_list = np.zeros(total_clusters)

    kappa_arr = lensing.get_convergence(ra_grid_deg, dec_grid_deg, ra_list, dec_list, M200c_list, redshift_list, param_dict)
    print('\tShape of convergence array is %s' %(str(kappa_arr.shape)))
    #imshow(kappa_arr[0]); colorbar(); show(); sys.exit()

sim_dic={}
if clusters_or_randoms == 'clusters': #cluster lensed sims
    do_lensing=True
    nclustersorrandoms=total_clusters
    sim_type='clusters'
elif clusters_or_randoms == 'randoms':
    do_lensing=False        
    nclustersorrandoms=total_randoms        
    sim_type='randoms'
    
sim_dic[sim_type]={}
sim_dic[sim_type]['sims'] = {}
print('\tcreating %s %s simulations' %(nclustersorrandoms, sim_type))
for simcntr in range( start, end ):
    print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))

    if random_seed_for_sims != -1:
        randomseedval = random_seed_for_sims * simcntr
        np.random.seed(randomseedval)

    sim_arr=[]
    for i in tqdm(range(nclustersorrandoms)):
        cmb_map=np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, cl[0])] )
        noise_map=np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])] )
        fg_map = np.zeros_like(noise_map)

        if do_lensing:
            cmb_map_lensed=[]
            unlensed_cmb=np.copy( cmb_map[0] )
            lensed_cmb=lensing.perform_lensing(ra_grid_deg, dec_grid_deg, unlensed_cmb, kappa_arr[i], mapparams)
            cmb_map_lensed.append( lensed_cmb )
            cmb_map=np.asarray(cmb_map_lensed)

        cmb_map = np.fft.ifft2( np.fft.fft2(cmb_map) * bl ).real
        fg_map = np.fft.ifft2( np.fft.fft2(fg_map) * bl ).real
        
        sim_map=cmb_map + noise_map + fg_map

        sim_map[tqu] -= np.mean(sim_map[0])
        sim_arr.append( sim_map )
    sim_dic[sim_type]['sims'][simcntr]=np.asarray( sim_arr )


sim_dic[sim_type]['cutouts_rotated'] = {}
sim_dic[sim_type]['grad_mag'] = {}
for simcntr in range( start, end ):
    print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))
    sim_arr=sim_dic[sim_type]['sims'][simcntr]
    nclustersorrandoms=len(sim_arr)
    if apply_wiener_filter:
        cl_signal_arr=[cl[0]]
        cl_noise_arr=[nl_dic['T']]


    grad_mag_arr, grad_orien_arr, cutouts_rotated_arr = tools.get_rotated_tqu_cutouts(sim_arr, 
                                                                                      sim_arr, 
                                                                                      nclustersorrandoms, 
                                                                                      tqulen, 
                                                                                      mapparams, 
                                                                                      cutout_size_am,
                                                                                      apply_wiener_filter=apply_wiener_filter, 
                                                                                      cl_signal = cl_signal_arr, 
                                                                                      cl_noise = cl_noise_arr,
                                                                                      lpf_gradient_filter = lpf_gradient_filter, 
                                                                                      cutout_size_am_for_grad =
                                                                                      cutout_size_am_for_grad)

    sim_dic[sim_type]['cutouts_rotated'][simcntr]=cutouts_rotated_arr
    sim_dic[sim_type]['grad_mag'][simcntr]=grad_mag_arr    


print('\tstack rotated cutouts + apply gradient magnitude weights')
sim_dic[sim_type]['stack'] = {}
for sim_type in sim_dic:
    for simcntr in range( start, end ):
        print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))
        cutouts_rotated_arr=sim_dic[sim_type]['cutouts_rotated'][simcntr]
        grad_mag_arr=sim_dic[sim_type]['grad_mag'][simcntr]

        stack = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)
        sim_dic[sim_type]['stack'][simcntr]=stack

        
fg_str=''
if fg_gaussian:
    fg_str = 'withgaussianfg'
else:
    fg_str = 'nogaussianfg'
if add_cluster_tsz:
    fg_str = '%s_withclustertsz' %(fg_str)
if add_cluster_ksz:
    fg_str = '%s_withclusterksz' %(fg_str)
fg_str = fg_str.strip('_')
mdef = 'm%s%s_%g' %(param_dict['delta'], param_dict['rho_def'], param_dict['cluster_mass'])
op_folder = misc.get_op_folder(results_folder, nx, dx, beamval, noiseval, cutout_size_am, mdef = mdef, ilc_file = ilc_file, which_ilc = which_ilc, nclustersorrandoms = total_clusters, pol = pol, fg_str = fg_str)
op_fname = misc.get_op_fname(op_folder, sim_type, nclustersorrandoms, end-start, start, end, random_seed_for_sims = random_seed_for_sims)
sim_dic[sim_type].pop('sims')
if clusters_or_randoms == 'randoms':
    sim_dic[sim_type].pop('cutouts_rotated')
    sim_dic[sim_type].pop('grad_mag')
sim_dic['param_dict']=param_dict
np.save(op_fname, sim_dic)
logline = 'All completed. Results dumped in %s' %(op_fname)
print(logline)

