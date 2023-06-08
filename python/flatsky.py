import numpy as np, sys, os, scipy as sc

################################################################################################################
#flat-sky routines
################################################################################################################

def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2D_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [ny, nx, dx] where ny, nx = flatskymap.shape; and dx is the pixel resolution in arcminutes.
    for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = 0.5 arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl, left = 0., right = 0.).reshape(ell.shape)

    return cl2d

################################################################################################################

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [ny, nx, dx] where ny, nx = flatskymap.shape; and dx is the pixel resolution in arcminutes.
    for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = 0.5 arcminutes.

    output:
    lx, ly
    """

    ny, nx, dx = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

################################################################################################################

def get_lxly_az_angle(lx,ly):

    """
    azimuthal angle from lx, ly

    inputs:
    lx, ly = 2d lx and ly arrays

    output:
    azimuthal angle
    """
    return 2*np.arctan2(lx, -ly)

################################################################################################################
def convert_eb_qu(map1, map2, flatskymapparams, eb_to_qu = 1):

    lx, ly = get_lxly(flatskymapparams)
    angle = get_lxly_az_angle(lx,ly)

    map1_fft, map2_fft = np.fft.fft2(map1),np.fft.fft2(map2)
    if eb_to_qu:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft - np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real
    else:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft + np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( -np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real

    return map1_mod, map2_mod
################################################################################################################

def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 0):
    """
    filter_type = 0 - low pass filter
    filter_type = 1 - high pass filter
    filter_type = 2 - band pass
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell>lmin_lmax] = 0.
    elif filter_type == 1:
        fft_filter[ell<lmin_lmax] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    fft_filter[np.isnan(fft_filter)] = 0.
    fft_filter[np.isinf(fft_filter)] = 0.

    return fft_filter
################################################################################################################

def wiener_filter(flatskymapparams, cl_signal, cl_noise, el = None):

    if el is None:
        el = np.arange(len(cl_signal))

    ny, nx, dx = flatskymapparams

    #get 2D cl
    cl_signal2d = cl_to_cl2d(el, cl_signal, flatskymapparams)
    cl_noise2d = cl_to_cl2d(el, cl_noise, flatskymapparams)

    wiener_filter = cl_signal2d / (cl_signal2d + cl_noise2d)
    badinds = np.where(cl_signal2d + cl_noise2d == 0.)
    wiener_filter[badinds] = 0.
    wiener_filter[np.isnan(wiener_filter)] = 0.
    wiener_filter[np.isinf(wiener_filter)] = 0.

    return wiener_filter

################################################################################################################

def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None, mask = None, filter_2d = None):

    """
    map2cl module - get the power spectra of map/maps

    input:
    flatskymyapparams = [ny, nx, dx] where ny, nx = flatskymap.shape; and dx is the pixel resolution in arcminutes.
    for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = 0.5 arcminutes.

    flatskymap1: map1 with dimensions (ny, nx)
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra

    binsize: el bins. computed automatically if None

    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2

    output:
    auto/cross power spectra: [el, cl, cl_err]
    """

    ny, nx, dx = flatskymapparams
    dx_rad = np.radians(dx/60.)

    lx, ly = get_lxly(flatskymapparams)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)

    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    if mask is not None:
        fsky = np.mean(mask)
        cl /= fsky

    if filter_2d is not None:
        rad_prf_filter_2d = radial_profile(filter_2d, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
        el, fl = rad_prf_filter_2d[:,0], rad_prf_filter_2d[:,1]
        cl /= fl

    return el, cl

################################################################################################################

def radial_profile(z, xy = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = 1):

    """
    get the radial profile of an image (both real and fourier space)
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(image.shape)
    else:
        x, y = xy

    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    if to_arcmins: radius *= 60.

    binarr=np.arange(minbin,maxbin,bin_size)
    radprf=np.zeros((len(binarr),3))

    hit_count=[]

    for b,bin in enumerate(binarr):
        ind=np.where((radius>=bin) & (radius<bin+bin_size))
        radprf[b,0]=(bin+bin_size/2.)
        hits = len(np.where(abs(z[ind])>0.)[0])

        if hits>0:
            radprf[b,1]=np.sum(z[ind])/hits
            radprf[b,2]=np.std(z[ind])
        hit_count.append(hits)

    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    errval=std_mean/(hit_count)**0.5
    radprf[:,2]=errval

    return radprf

################################################################################################################

def make_gaussian_realisation(mapparams, el, cl, cl2 = None, cl12 = None, cltwod=None, tf=None, bl = None, qu_or_eb = 'qu'):

    ny, nx, dx = mapparams
    arcmins2radians = np.radians(1/60.)

    dx *= arcmins2radians

    ################################################
    #map stuff
    norm = np.sqrt(1./ (dx**2.))
    ################################################

    #if cltwod is given, directly use it, otherwise do 1d to 2d
    if cltwod is None:
        cltwod = cl_to_cl2d(el, cl, mapparams)

    # if the tranfer function is given, correct the 2D cl by tf
    if tf is not None:
        if isinstance(tf, np.ndarray):
            cltwod = cltwod * tf**2
        else:
            cltwod = cltwod * tf['T']**2

    ################################################
    if cl2 is not None: #for TE, etc. where two fields are correlated.
        assert cl12 is not None
        cltwod12 = cl_to_cl2d(el, cl12, mapparams)
        cltwod2 = cl_to_cl2d(el, cl2, mapparams)
        if tf is not None:
            cltwod2 = cltwod2 * tf['E']**2
            cltwod12 = cltwod12 * tf['T'] * tf['E']

    ################################################
    if cl2 is None:

        cltwod = cltwod**0.5 * norm
        cltwod[np.isnan(cltwod)] = 0.

        gauss_reals = np.random.standard_normal([nx,ny])
        SIM = np.fft.ifft2( np.copy( cltwod ) * np.fft.fft2( gauss_reals ) ).real

    else: #for TE, etc. where two fields are correlated.

        assert qu_or_eb in ['qu', 'eb']

        cltwod[np.isnan(cltwod)] = 0.
        cltwod12[np.isnan(cltwod12)] = 0.
        cltwod2[np.isnan(cltwod2)] = 0.

        #in this case, generate two Gaussian random fields
        #SIM_FIELD_1 will simply be generated from gauss_reals_1 like above
        #SIM_FIELD_2 will generated from both gauss_reals_1, gauss_reals_2 using the cross spectra
        gauss_reals_1 = np.random.standard_normal([nx,ny])
        gauss_reals_2 = np.random.standard_normal([nx,ny])

        gauss_reals_1_fft = np.fft.fft2( gauss_reals_1 )
        gauss_reals_2_fft = np.fft.fft2( gauss_reals_2 )

        #field_1
        cltwod_tmp = np.copy( cltwod )**0.5 * norm
        SIM_FIELD_1 = np.fft.ifft2( cltwod_tmp *  gauss_reals_1_fft ).real
        #SIM_FIELD_1 = np.zeros( (ny, nx) )

        #field 2 - has correlation with field_1
        t1 = np.copy( gauss_reals_1_fft ) * cltwod12 / np.copy(cltwod)**0.5
        t2 = np.copy( gauss_reals_2_fft ) * ( cltwod2 - (cltwod12**2. /np.copy(cltwod)) )**0.5
        SIM_FIELD_2_FFT = (t1 + t2) * norm
        SIM_FIELD_2_FFT[np.isnan(SIM_FIELD_2_FFT)] = 0.
        SIM_FIELD_2 = np.fft.ifft2( SIM_FIELD_2_FFT ).real

        #T and E generated. B will simply be zeroes.
        SIM_FIELD_3 = np.zeros( SIM_FIELD_2.shape )
        if qu_or_eb == 'qu' and pol_fields: #T, Q, U: convert E/B to Q/U.
            SIM_FIELD_2, SIM_FIELD_3 = convert_eb_qu(SIM_FIELD_2, SIM_FIELD_3, mapparams, eb_to_qu = 1)
        else: #T, E, B: B will simply be zeroes
            pass

        SIM = np.asarray( [SIM_FIELD_1, SIM_FIELD_2, SIM_FIELD_3] )

    if bl is not None:
        if np.ndim(bl) != 2:
            bl = cl_to_cl2d(el, bl, mapparams)
        SIM = np.fft.ifft2( np.fft.fft2(SIM) * bl).real

    if cl2 is None:
        SIM = SIM - np.mean(SIM)
    else:
        for tqu in range(len(SIM)):
            SIM[tqu] = SIM[tqu] - np.mean(SIM[tqu])

    return SIM

################################################################################################################
