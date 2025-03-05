import numpy as np, sys, os, scipy as sc, argparse
sys_path_folder='../python/'
sys.path.append(sys_path_folder)

import flatsky, tools, lensing, foregrounds, misc

from tqdm import tqdm

from pylab import *
cmap = cm.RdYlBu_r

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)

#==============================================================================================================================#
parser = argparse.ArgumentParser(description='')
parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='../results//nx120_dx1/beam1.2/noise5/10amcutouts/nogaussianfg/T/clusters_700objects_10sims0to10.npy')
parser.add_argument('-minM', dest='minM', action='store', help='minM', type=float, default=0.)
parser.add_argument('-maxM', dest='maxM', action='store', help='maxM', type=float, default=4.)
parser.add_argument('-delM', dest='delM', action='store', help='delM', type=float, default=0.1)
parser.add_argument('-totiters_for_model', dest='totiters_for_model', action='store', help='totiters_for_model', type=int, default=1)#25)
parser.add_argument('-random_seed_for_models', dest='random_seed_for_models', action='store', help='random_seed_for_models', type=int, default=100)

args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]

    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

#==============================================================================================================================#
data = np.load(dataset_fname, allow_pickle= True).item()
param_dict = data['param_dict']
for key, value in param_dict.items():
    globals()[key] = value
    
nx = int(boxsize_am/dx)
mapparams = [nx, nx, dx]
x1,x2 = -nx/2. * dx, nx/2. * dx
verbose = 0
pol = param_dict['pol']
debug = param_dict['debug']

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
########################

########################
#get beam and noise
bl = tools.get_bl(beamval, el, make_2d = 1, mapparams = mapparams)
nl_dic = tools.get_nl_dic(noiseval, el, pol = pol)
do_lensing=True
nclustersorrandoms=total_clusters
sim_type='clusters'
ra_grid_deg, dec_grid_deg = ra_grid/60., dec_grid/60.


cluster_mass_arr = np.arange( minM, maxM+delM/10., delM ) * 1e14
cluster_z_arr = np.tile( cluster_z, len(cluster_mass_arr) )

for (cluster_mass, cluster_z) in zip(cluster_mass_arr, cluster_z_arr):
    keyname = (round(cluster_mass/1e14, 3), round(cluster_z, 3))
    print('\t###############')
    print('\tcreating model for %s' %(str(keyname)))
    print('\t###############')


    np.random.seed( random_seed_for_models )
    print('\t\tsetting random seed for model generation. seed is %s' %(random_seed_for_models))

    M200c_list = np.tile(cluster_mass, total_clusters)
    redshift_list = np.tile(cluster_z, total_clusters)
    ra_list = dec_list = np.zeros(total_clusters)

    kappa_arr = lensing.get_convergence(ra_grid_deg, dec_grid_deg, ra_list, dec_list, M200c_list, redshift_list, param_dict)


    sim_dic={}
    sim_dic[sim_type]={}
    sim_dic[sim_type]['sims'] = {}
    sim_dic[sim_type]['cmb_sims'] = {}
    print('\t\tcreating %s %s simulations' %(nclustersorrandoms, sim_type))
    for simcntr in range( totiters_for_model ):
        print('\t\t\tmodel dataset %s of %s' %(simcntr+1, totiters_for_model))
        cmb_sim_arr,sim_arr=[],[]
        #for i in tqdm(range(nclustersorrandoms)):
        for i in range(nclustersorrandoms):
            if not pol:
                cmb_map=np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, cl[0])] )
                noise_map=np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])] )
                if fg_gaussian:
                    fg_map=np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, cl_fg_dic['T'])] )
                else:
                    fg_map = np.zeros_like(noise_map)
            
            if do_lensing:
                cmb_map_unlensed = np.copy(cmb_map)
                cmb_map_lensed=[]

                unlensed_cmb=np.copy( cmb_map[0] )
                lensed_cmb=lensing.perform_lensing(ra_grid_deg, dec_grid_deg, unlensed_cmb, kappa_arr[i], mapparams)

                cmb_map_lensed.append( lensed_cmb )
                cmb_map=np.asarray(cmb_map_lensed)

            #add beam
            cmb_map = np.fft.ifft2( np.fft.fft2(cmb_map) * bl ).real
            fg_map = np.fft.ifft2( np.fft.fft2(fg_map) * bl ).real
                
            sim_map=np.copy(cmb_map)+noise_map+fg_map

            sim_map[0] -= np.mean(sim_map[0])
            cmb_map[0] -= np.mean(cmb_map[0])

            sim_arr.append( sim_map )
            cmb_sim_arr.append( cmb_map )
        sim_dic[sim_type]['sims'][simcntr]=np.asarray( sim_arr )
        sim_dic[sim_type]['cmb_sims'][simcntr]=np.asarray( cmb_sim_arr )

        
    print('\t\tget gradient information for all cluster cutouts')

    sim_dic[sim_type]['cutouts_rotated'] = {}
    sim_dic[sim_type]['grad_mag'] = {}
    sim_dic[sim_type]['grad_orien'] = {}


    sim_arr=sim_dic[sim_type]['sims'][0]
    cmb_sim_arr=sim_dic[sim_type]['cmb_sims'][0]
    nclustersorrandoms=len(sim_arr)
    
    if apply_wiener_filter:
        cl_signal_arr=[cl[0]]
        cl_noise_arr=[nl_dic['T']]


    grad_mag_arr, grad_orien_arr, cutouts_rotated_arr = tools.get_rotated_tqu_cutouts(cmb_sim_arr, 
                                                                                      sim_arr, 
                                                                                      nclustersorrandoms, 
                                                                                      tqulen, 
                                                                                      mapparams, 
                                                                                      cutout_size_am, 
                                                                                      perform_rotation = False,
                                                                                      apply_wiener_filter=apply_wiener_filter, 
                                                                                      cl_signal = cl_signal_arr, 
                                                                                      cl_noise = cl_noise_arr,
                                                                                      lpf_gradient_filter = lpf_gradient_filter, 
                                                                                      cutout_size_am_for_grad =
                                                                                      cutout_size_am_for_grad)

        sim_dic[sim_type]['cutouts_rotated'][0]=cutouts_rotated_arr
        sim_dic[sim_type]['grad_mag'][0]=grad_mag_arr
        sim_dic[sim_type]['grad_orien'][0]=grad_orien_arr



    model_dic = {}


    model_dic[simcntr] = {}        
    print('\t\t\tmodel dataset %s of %s' %(simcntr+1, totiters_for_model))
    cutouts_rotated_arr=sim_dic[sim_type]['cutouts_rotated'][0]
    grad_mag_arr=sim_dic[sim_type]['grad_mag'][0]
    grad_orien_arr=sim_dic[sim_type]['grad_orien'][0]

    stack=tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)

    model_dic[0]['stack'] = stack
    model_dic[0]['cutouts'] = [cutouts_rotated_arr, grad_mag_arr, grad_orien_arr]


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
    op_folder = misc.get_op_folder(results_folder, nx, dx, beamval, noiseval, cutout_size_am, mdef=mdef, ilc_file=ilc_file, which_ilc = which_ilc, nclustersorrandoms = total_clusters, pol = pol, models = True, fg_str = fg_str)
    extrastr = '_randomseed%s_mass%.3f_z%.3f' %(random_seed_for_models, keyname[0], keyname[1])
    op_fname = misc.get_op_fname(op_folder, sim_type, nclustersorrandoms, totiters_for_model, extrastr = extrastr)
    np.save(op_fname, model_dic)
    logline = '\t\tResults dumped in %s' %(op_fname)
    print(logline)
sys.exit()
