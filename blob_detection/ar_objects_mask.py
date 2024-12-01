import numpy as np
import xarray as xr
import datetime as dt
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve, minimum_filter

import os


def reorder_image_labels(label_im):
    label_im_new = label_im.copy()
    unique_labels = sorted(np.unique(label_im))
    for n, label in enumerate(unique_labels):
        if n > 0:
            label_im_new[label_im == label] = n
    return (label_im_new, len(unique_labels)-1)

def ndimage_label_periodic_x(image):
    label_im, labels = ndimage.label(image)

    more_changes = True
    while more_changes:
        more_changes = False
        west_edge_ids = []
        east_edge_ids = []
        for n in range(1, labels+1):
            # Check left side.
            this_feature = label_im == n
            if np.sum(this_feature[:,0]) > 0:
                west_edge_ids += [n]
            # Check right side.
            if np.sum(this_feature[:,-1]) > 0:
                east_edge_ids += [n]

        # Check whether the western and eastern edge features should be combined.
        changes = 0
        for n in west_edge_ids:
            this_feature_n = label_im == n
            for m in east_edge_ids:
                if not m == n: # Check if already the same id.
                    this_feature_m = label_im == m
                    if np.nansum(np.logical_and(this_feature_n[:,0], this_feature_m[:,-1])):
                        # I have a match! So set the east edge system to the west edge ID.
                        label_im[this_feature_m] = n
                        changes += 1

        if changes > 0:
            label_im, labels = reorder_image_labels(label_im)
            more_changes = True

    return (label_im, labels)


class ar_object_mask:
    # Initialize empty masks and supporting variables.
    # Initialize using lat and lon coordinates.
    def __init__(self, valid_datetime):
        
        self.time = valid_datetime

        ## Initialize arrays.
        # The AR mask
        self.mask = np.array([], dtype='bool')
        self.mask_ivt = np.array([], dtype='bool')
        self.mask_tpw = np.array([], dtype='bool')

        self.lon = np.array([])
        self.lat = np.array([])
        self.grid_cell_area = np.array([])
        self.topography = np.array([])

        ## Place holders for the supporting variables
        ## that go along with the mask.

        # TPW variables
        self.tpw = np.array([])
        self.tpw_background = np.array([])
        self.max_tpw = np.array([])
        self.mean_tpw = np.array([])
        self.deep_tropics_mask = np.array([])

        # IVT variables
        self.ivt = np.array([])
        self.ivt_background = np.array([])
        self.max_ivt = np.array([])
        self.mean_ivt = np.array([])
        self.ivt_eastward = np.array([])
        self.ivt_northward = np.array([])


    def read(self, path_fmt = './mask_data/ar%Y%m%d%H.nc', verbose=True):
        """
        Read in AR mask data from an existing NetCDF file.
        """
        fn = self.time.strftime(path_fmt)
        if verbose:
            print(f'Reading in AR mask data from {fn}.')
            
        with xr.open_dataset(fn) as ds:
            ## Account for the possibility of 3-D lat/lon,
            ## Where the first dimension is time.
            s = ds['lon'].data.shape
            if (len(s) == 3 and s[0] == 1):
                self.set_coordinates(ds['lon'].data[0,:,:],
                                     ds['lat'].data[0,:,:])
            else:
                # Note: This here will throw an exception
                #       if the dimensions are otherwise weird.
                self.set_coordinates(ds['lon'].data, ds['lat'].data)

            for field in ['grid_cell_area','tpw','tpw_background','max_tpw',
                          'mean_tpw','deep_tropics_mask',
                          'ivt','ivt_eastward','ivt_northward','ivt_background',
                          'max_ivt','mean_ivt','mask','mask_ivt','mask_tpw']:
                self.__set_2d_field(field, ds[field].data[0,:,:])
    

    def write(self, path_fmt = './mask_data/ar%Y%m%d%H.nc',
              verbose=True, full_output=False):
        """
        This method writes the AR mask data to a NetCDF file.
        """
        fn = self.time.strftime(path_fmt)
        if verbose:
            print('Saving AR mask to: '+fn)

        # make sure all times get stored as "seconds since 1970-1-1"
        timestamp = (self.time - dt.datetime(1970,1,1,0,0,0)).total_seconds()

        # Build an xarray Dataset for output to NetCDF.
        coords = {'time': (['time',], [timestamp], {'units':'seconds since 1970-1-1'}),
                  'lon': (['lon',], self.lon[0,:], {'units':'degrees_East'}),
                  'lat': (['lat',], self.lat[:,0], {'units':'degrees_North'})}
        data_vars = {}
        for field in ['mask','mask_ivt','mask_tpw','deep_tropics_mask']:
            data_vars[field] = (['time','lat','lon'], np.expand_dims(getattr(self, field),0),
                              {'units':'1'})

        if full_output:
            for field in ['grid_cell_area',]:
                data_vars[field] = (['lat'], getattr(self, field)[:,0],
                            {'units':'kg2'})
            for field in ['topography',]:
                data_vars[field] = (['time','lat','lon'], np.expand_dims(getattr(self, field),0),
                                {'units':'m'})

        if full_output:
            for field in ['tpw','tpw_background']: #,'max_tpw','mean_tpw']:
                data_vars[field] = (['time','lat','lon'], np.expand_dims(getattr(self, field),0),
                                {'units':'mm'})
            for field in ['ivt',]: #'ivt_background','ivt_eastward','ivt_northward','max_ivt','mean_ivt']:
                data_vars[field] = (['time','lat','lon'], np.expand_dims(getattr(self, field),0),
                                {'units':'kg m-1 s-1'})
            
        ds_out = xr.Dataset(data_vars=data_vars, coords=coords)

        # Write out the NetCDF file.
        encoding = {}
        for field in ['mask','mask_ivt','mask_tpw','deep_tropics_mask']:
            if field in data_vars:
                encoding[field] = {'dtype':'i1', 'zlib':True}
        for field in ['lon','lat','grid_cell_area','topography','tpw','tpw_background','max_tpw',
                      'ivt','ivt_background','ivt_eastward','ivt_northward','max_ivt']:
            if field in data_vars:
                encoding[field] = {'dtype':'float32', 'zlib':True}

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        ds_out.to_netcdf(fn, encoding=encoding)

    def set_coordinates(self, lon, lat, verbose=True):
        # If 1D coordinates, make them be 2D
        # Otherwise just use the 2D coordinates.
        # Throw an exception if neither 2D nor 3D.
        if lon.ndim == 1 and lat.ndim == 1:
            if verbose:
                print('Converting 1D lon, lat to 2D.')
            self.lon, self.lat = np.meshgrid(lon, lat)
        elif lon.ndim == 2 and lat.ndim == 2:
            if verbose:
                print('Using 2D lon, lat data.')
            self.lon = lon; self.lat = lat;
        else:
            raise Exception(
                'Invalid lon and lat dimensions. Use either 1-D or 2-D arrays.'
                )

        dx = np.gradient(self.lon, axis=1)
        dy = np.gradient(self.lat, axis=0)
        self.__set_2d_field(
            'grid_cell_area',
            np.abs(111.0*111.0*dx*dy * np.cos(self.lat * np.pi / 180.0))
            )

    def are_coordinates_set(self):
        if self.lon.ndim == 2 and self.lat.ndim == 2:
            return True
        else:
            return False

    def set_tpw(self, tpw):
        self.__set_2d_field('tpw', tpw)

    def set_topography(self, topography):
        self.__set_2d_field('topography', topography)

    def set_ivt(self, viwve_data, viwvn_data):
        self.__set_2d_field('ivt_eastward', viwve_data)
        self.__set_2d_field('ivt_northward', viwvn_data)
        self.__set_2d_field(
            'ivt',
            np.sqrt(np.power(self.ivt_eastward, 2)
                    + np.power(self.ivt_northward, 2))
            )

    def calc_tpw_background_old(self, smooth=True):
        
        # Set the kernel used for the spatial filter
        kernel = np.ones([11,161])
        kernel = kernel / np.sum(kernel)

        kernel_lat = np.ones([161,11])
        kernel_lat = kernel_lat / np.sum(kernel_lat)

        if self.tpw.ndim == 2:

            # I'm using "wrap" mode here, but I actually only want it to wrap in the 
            # zonal direction. So I pad it in the meridional direction
            # Then remove the artificial points at the north and south edges.

            # Longitudinal 
            n_pad_y = kernel.shape[0]
            tpw_padded = np.append(self.tpw[0:n_pad_y,:], self.tpw, axis=0)
            tpw_padded = np.append(tpw_padded, self.tpw[-1*n_pad_y:,:], axis=0)

            tpw_background = convolve(tpw_padded, kernel, mode='wrap')
            tpw_background = tpw_background[n_pad_y:-1*n_pad_y,:]

            self.tpw_background = tpw_background

            # Latitudinal
            n_pad_y_lat = kernel_lat.shape[0]
            tpw_padded_lat = np.append(self.tpw[0:n_pad_y_lat,:], self.tpw, axis=0)
            tpw_padded_lat = np.append(tpw_padded_lat, self.tpw[-1*n_pad_y_lat:,:], axis=0)

            tpw_background_lat = convolve(tpw_padded_lat, kernel_lat, mode='wrap')
            tpw_background_lat = tpw_background_lat[n_pad_y_lat:-1*n_pad_y_lat,:]

            self.tpw_background_lat = tpw_background_lat

            # adjust the background over high topography.
            # I'm not doing this yet, but here was my first try:
            # It should really be adjusted based on the standard atmosphere
            # or some other kind of profile.
            # self.tpw_background = self.tpw_background * (1.0 - self.topography/10000.0)
        else:
            raise Exception('Set TPW before calling calc_tpw_threshold.')


    def calc_tpw_background(self):
        
        rad = int(5.0/.25)
        MAXTOPO = 500.0
        # MAXTOPO = 1000.0

        if self.tpw.ndim == 2:

            n_pad_y_lat = 1*rad
            tpw_padded_lat = np.append(self.tpw[0:n_pad_y_lat,:], self.tpw, axis=0)
            tpw_padded_lat = np.append(tpw_padded_lat, self.tpw[-1*n_pad_y_lat:,:], axis=0)
            topo_padded_lat = np.append(self.topography[0:n_pad_y_lat,:], self.topography, axis=0)
            topo_padded_lat = np.append(topo_padded_lat, self.topography[-1*n_pad_y_lat:,:], axis=0)

            #Effectively disregard points above the MAXTOPO elevation.
            tpw_topo_max = np.where(topo_padded_lat > MAXTOPO, 999.0, tpw_padded_lat)

            # Set the shapes used for the minimum filters
            shape_left = np.append(np.ones([1,rad]), np.zeros([1,rad+1]), axis=1)
            shape_right = np.append(np.zeros([1,rad+1]), np.ones([1,rad]), axis=1)
            shape_top = np.append(np.ones([rad,1]), np.zeros([rad+1,1]), axis=0)
            shape_bottom = np.append(np.zeros([rad+1,1]), np.ones([rad,1]), axis=0)

            # Calculate the minimum filters.
            # I'm using "wrap" mode here, but I actually only want it to wrap in the 
            # zonal direction. So I pad it in the meridional direction
            # Then remove the artificial points at the north and south edges.
            tpw_filter_left = minimum_filter(tpw_topo_max, footprint=shape_left, mode='wrap')
            tpw_filter_right = minimum_filter(tpw_topo_max, footprint=shape_right, mode='wrap')
            tpw_filter_top = minimum_filter(tpw_topo_max, footprint=shape_top, mode='wrap')
            tpw_filter_bottom = minimum_filter(tpw_topo_max, footprint=shape_bottom, mode='wrap')

            # Combine the minimum filters to get the full background TPW.
            tpw_filter_left_right = np.maximum(tpw_filter_left, tpw_filter_right)
            tpw_filter_top_bottom = np.maximum(tpw_filter_top, tpw_filter_bottom)

            tpw_filter_all = np.minimum(tpw_filter_left_right, tpw_filter_top_bottom)

            # Throw away the data where I stitched at the north and south edges.
            tpw_filter_all = tpw_filter_all[n_pad_y_lat:-1*n_pad_y_lat,:]

            # I don't have a background TPW for high topography.
            # Therefore, set it to the original TPW.
            # That way when I calculate the difference from the background,
            # I will get zero.
            tpw_filter_all = np.where(tpw_filter_all > 998.0, self.tpw, tpw_filter_all)

            # Set the background TPW.
            self.tpw_background = tpw_filter_all



        else:
            raise Exception('Set TPW before calling calc_tpw_threshold.')


    def calc_ivt_background(self, smooth=True):
        
        if self.ivt.ndim == 2:

            ## Oceanic IVT
            ivt_oceanic = 1.0*self.ivt
            # Don't block out Antarctica, where there is no oceanic IVT.
            ivt_oceanic[np.logical_and(self.topography > 1.0, self.lat > -60.0)] = np.nan

            # Use zero for Antarctica, where there is no oceanic IVT.
            # ivt_oceanic[np.isnan(ivt_oceanic)] = 0.0

            # Zonal mean oceanic, Indo-Pacific (longitude: 80 - 260)
            ivt_oceanic_pac = 1.0*ivt_oceanic
            ivt_oceanic_pac[self.lon < 140] = np.nan
            ivt_oceanic_pac[self.lon > 260] = np.nan

            zonal_mean_ivt_oceanic_pac = np.nanmean(ivt_oceanic_pac, axis=1) 

            dum, zonal_mean_ivt_oceanic_pac_2d = np.meshgrid(
                self.lon[0,:], zonal_mean_ivt_oceanic_pac)

            ## Cap value outside the tropics
            ## because the latitudes with ARs will have higher IVT.
            zonal_mean_ivt_oceanic_pac_2d[self.lat > 20.0] = np.minimum(
                zonal_mean_ivt_oceanic_pac[self.lat[:,0] == 20],
                zonal_mean_ivt_oceanic_pac_2d[self.lat > 20.0])
            zonal_mean_ivt_oceanic_pac_2d[self.lat < -20.0] = np.minimum(
                zonal_mean_ivt_oceanic_pac[self.lat[:,0] == -20],
                zonal_mean_ivt_oceanic_pac_2d[self.lat < -20.0])

            zonal_mean_ivt_oceanic_pac_2d = ndimage.convolve1d(
                zonal_mean_ivt_oceanic_pac_2d,
                np.ones(41)/41,
                mode='nearest',
                axis=0)

            self.ivt_background = zonal_mean_ivt_oceanic_pac_2d


        else:
            raise Exception('Set IVT before calling calc_ivt_threshold.')


    def calc_deep_tropics_mask_tpw(self, verbose=False):

        """
        calc_deep_tropics_mask(self, verbose=False)

        Determine the deep tropics mask based on high TPW in the deep tropics.

        Use the erosion - dilation method to cut off the "tentacles" which
        are themselves ARs.
        """

        # 1. First guess mask: High TPW.
        # - 35 mm was way into the extra tropics.
        # - 45 mm only really picks up the Indo-Pacific region.
        deep_tropics_mask0 = (self.tpw > 40.0)

        # 2. Erosion to eliminate the "tentacles" that stick out (e.g., ARs)

        # 2.1. Shape used for erosion/dilation: A circle of radius "shape_radius"
        dx = self.lon[0,1] - self.lon[0,0]
        shape_radius = int(5.0 / dx)
        shape = np.zeros([2*shape_radius+1, 2*shape_radius+1])
        x, y = np.meshgrid(
            np.arange(-1*shape_radius, shape_radius+1),
            np.arange(-1*shape_radius, shape_radius+1))
        shape_dist = np.sqrt(np.power(x,2) + np.power(y,2))
        shape[shape_dist <= shape_radius] = 1

        # 2.2. Do the erosion.
        deep_tropics_mask_erosion = ndimage.binary_erosion(deep_tropics_mask0, shape)

        # 3. Narrow ITCZ might be broken up by 1. Get it back.
        for ii in range(len(self.tpw[0,:])):
            this_tpw_col = 1.0*self.tpw[:,ii]
            this_tpw_col[np.abs(self.lat[:,0]) > 10] = 0.0
            deep_tropics_mask_erosion[np.argmax(this_tpw_col),ii] = 1

        # 4. Expand the area to get back close to the original contour.
        deep_tropics_mask = ndimage.binary_dilation(deep_tropics_mask_erosion, shape)

        # 5. Fill in holes.
        deep_tropics_mask = ndimage.binary_fill_holes(deep_tropics_mask)

        # 6. Eliminate blobs that are too small to be deep tropics (planetary scale)
        mask_labeled, nb_labels = ndimage_label_periodic_x(deep_tropics_mask)
        for n in range(1,nb_labels+1):
            if np.nansum(self.grid_cell_area * (mask_labeled==n)) < 1e7:
                deep_tropics_mask[mask_labeled == n] = 0

        # 7. Taking out the concave "kinks" using the binary_closing method
        #    binary_closing is a dilation then an erosion.
        n_pad_x = shape.shape[0]
        deep_tropics_mask_padded = np.append(deep_tropics_mask[:,0:n_pad_x], deep_tropics_mask, axis=1)
        deep_tropics_mask_padded = np.append(deep_tropics_mask_padded, deep_tropics_mask[:,-1*n_pad_x:], axis=1)
        deep_tropics_mask_padded = ndimage.binary_closing(deep_tropics_mask_padded, structure=shape)
        deep_tropics_mask = deep_tropics_mask_padded[:,n_pad_x:-1*n_pad_x]

        # 8. Finally, set the class object's deep_tropics_mask property.
        self.deep_tropics_mask = deep_tropics_mask



    def calc_deep_tropics_mask(self, verbose=False):

        """
        calc_deep_tropics_mask(self, verbose=False)

        Determine the deep tropics mask based on high TPW in the deep tropics.

        Use the erosion - dilation method to cut off the "tentacles" which
        are themselves ARs.
        """

        # 1. First guess mask: Based on edge detection of TPW (Laplacian of Gaussian)
        tpw_LoG_filtered = ndimage.gaussian_laplace(self.tpw, [5,5], truncate=2)

        # Attempt an alternative deep tropics mask
        deep_tropics_mask0 = (tpw_LoG_filtered < -0.5)

        # 2. Erosion to eliminate the "tentacles" that stick out (e.g., ARs)

        # 2.1. Shape used for erosion/dilation: A circle of radius "shape_radius"
        dx = self.lon[0,1] - self.lon[0,0]
        shape_radius = int(5.0 / dx)
        shape = np.zeros([2*shape_radius+1, 2*shape_radius+1])
        x, y = np.meshgrid(
            np.arange(-1*shape_radius, shape_radius+1),
            np.arange(-1*shape_radius, shape_radius+1))
        shape_dist = np.sqrt(np.power(x,2) + np.power(y,2))
        shape[shape_dist <= shape_radius] = 1

        # 2.2. Do the erosion.
        deep_tropics_mask_erosion = ndimage.binary_erosion(deep_tropics_mask0, shape)

        # 3. Narrow ITCZ might be broken up by 1. Get it back.
        for ii in range(len(self.tpw[0,:])):
            this_tpw_col = 1.0*self.tpw[:,ii]
            this_tpw_col[np.abs(self.lat[:,0]) > 10] = 0.0
            deep_tropics_mask_erosion[np.argmax(this_tpw_col),ii] = 1
            deep_tropics_mask_erosion[this_tpw_col > 50.0,ii] = 1

        # 4. Expand the area to get back close to the original contour.
        deep_tropics_mask = ndimage.binary_dilation(deep_tropics_mask_erosion, shape)

        # 5. Fill in holes.
        deep_tropics_mask = ndimage.binary_fill_holes(deep_tropics_mask)

        # 6. Eliminate blobs that are too small to be deep tropics (planetary scale)
        mask_labeled, nb_labels = ndimage_label_periodic_x(deep_tropics_mask)
        for n in range(1,nb_labels+1):
            if np.nansum(self.grid_cell_area * (mask_labeled==n)) < 1e7:
                deep_tropics_mask[mask_labeled == n] = 0

        # 7. Taking out the concave "kinks" using the binary_closing method
        #    binary_closing is a dilation then an erosion.
        n_pad_x = shape.shape[0]
        deep_tropics_mask_padded = np.append(deep_tropics_mask[:,0:n_pad_x], deep_tropics_mask, axis=1)
        deep_tropics_mask_padded = np.append(deep_tropics_mask_padded, deep_tropics_mask[:,-1*n_pad_x:], axis=1)
        deep_tropics_mask_padded = ndimage.binary_closing(deep_tropics_mask_padded, structure=shape)
        deep_tropics_mask = deep_tropics_mask_padded[:,n_pad_x:-1*n_pad_x]

        # 8. Finally, set the class object's deep_tropics_mask property.
        self.deep_tropics_mask = deep_tropics_mask



    def calc_ar_mask(self, verbose=False):

        """
        calc_ar_mask(self, verbose=False)

        This is the main function for calculating the AR mask.

        1. For IVT: Use Laplacian of Gaussian filter on the IVT anomaly,
        #  and threshold of IVT_GAUSS_LAPLACE_THRESH.
        2. For TPW: Take the difference from the background TPW field.
        3. AR mask satisfies either IVT or TPW criteria. Combined feature MUST
           have an area > MINAREA that satisfies each of the criteria.
        4. Feature must stick out of the deep_tropics_mask by at least MINAREA.
        5. Features are not allowed to cross the equator.
        """
        # IVT_GAUSS_LAPLACE_THRESH = -10.0
        IVT_GAUSS_LAPLACE_THRESH = -7.0
        TPW_ANOM_THRESH = 10.0
        MINAREA = 120000.0

        # if self.tpw_background.ndim == 2 and self.ivt_background.ndim == 2:
        if self.deep_tropics_mask.ndim == 2 and self.ivt.ndim == 2:

            self.mask = 0.0 * self.ivt
            self.max_tpw = 0.0 * self.ivt
            self.max_ivt = 0.0 * self.ivt
            area = 1.0*self.grid_cell_area

            # 1. Get IVT Mask
            # Use Laplacian of Gaussian filter, and threshold of IVT_GAUSS_LAPLACE_THRESH

            # Apply Laplace of Gaussian filter.
            ivt_filtered = ndimage.gaussian_laplace(self.ivt, [5,5], truncate=2)

            # Mask satisfies the threshold.
            mask_ivt = (ivt_filtered < IVT_GAUSS_LAPLACE_THRESH)

            # Nerf the strong monsoon circulations:
            # - Eastward IVT > 500
            # - Within deep tropics mask
            mask_ivt[np.logical_and(
                self.deep_tropics_mask,
                self.ivt_eastward > 250.0
                )] = 0
            mask_ivt[np.logical_and(
                self.deep_tropics_mask,
                self.ivt_eastward < -250.0
                )] = 0

            mask_ivt = ndimage.binary_fill_holes(mask_ivt)

            self.mask_ivt = mask_ivt


            # 2. Get TPW Mask
            #    - Outside the deep_tropics_mask: Use the longitude + latitude box filters.
            #    - Inside the deep tropics mask: Use only the longitude box filter.

            mask_tpw = np.logical_or(
                self.tpw - self.tpw_background > TPW_ANOM_THRESH,
                self.tpw - self.tpw_background > 0.5*self.tpw_background
            )
                

            """
            thresh = 1*TPW_ANOM_THRESH
            rad = int(5.0/.25)
            # rad = int(7.5/.25)

            max_topo_for_tpw = 500.0 # meters
            # max_topo_for_tpw = 1000.0 # meters  # Still picking up TPW topo gradients when I use 1000 m.

            is_local_max = 0.0 * self.tpw
            dt11 = dt.datetime.now()

            S = self.tpw.shape
            # nskip = 0
            for jj in range(rad, S[0] - rad):
                for ii in range(rad, S[1] - rad):
                    if (self.topography[jj,ii] > max_topo_for_tpw):
                        continue

                    if jj < rad:
                        this_col = np.append(np.full(rad-jj, self.tpw[0,ii]), self.tpw[0:jj+rad+1, ii])
                        this_col_topo = np.append(np.full(rad-jj, self.topography[0,ii]), self.topography[0:jj+rad+1, ii])
                    elif jj > S[0]-rad-1:
                        this_col = np.append(self.tpw[jj-rad:S[0], ii], np.full(rad-(S[0]-jj), self.tpw[S[0]-1,ii]))
                        this_col_topo = np.append(self.topography[jj-rad:S[0], ii], np.full(rad-(S[0]-jj), self.topography[S[0]-1,ii]))
                    else:
                        this_col = 1.0*self.tpw[jj-rad:jj+rad+1, ii]
                        this_col_topo = self.topography[jj-rad:jj+rad+1, ii]
                    this_col[this_col_topo > max_topo_for_tpw] = 999

                    min_val_1 = np.nanmin(this_col[0:rad])
                    min_val_2 = np.nanmin(this_col[rad+1:])
                    diff1 = self.tpw[jj,ii] - min_val_1
                    diff2 = self.tpw[jj,ii] - min_val_2
                    if (diff1 > thresh and diff2 > thresh):
                        is_local_max[jj,ii] = 1
                        # nskip += 1
                        # continue

                    if ii < rad:
                        this_row = np.append(self.tpw[jj, S[1]-(rad-ii):S[1]], self.tpw[jj, 0:ii+rad+1])
                        this_row_topo = np.append(self.topography[jj, S[1]-(rad-ii):S[1]], self.topography[jj, 0:ii+rad+1])
                    elif ii > S[1]-rad-1:
                        this_row = np.append(self.tpw[jj, ii-rad:S[1]], self.tpw[jj, 0:rad-(S[1]-ii-1)])
                        this_row_topo = np.append(self.topography[jj, ii-rad:S[1]], self.topography[jj, 0:rad-(S[1]-ii-1)])
                    else:
                        this_row = 1.0*self.tpw[jj, ii-rad:ii+rad+1]
                        this_row_topo = self.topography[jj, ii-rad:ii+rad+1]
                    this_row[this_row_topo > max_topo_for_tpw] = 999

                    min_val_1 = np.nanmin(this_row[0:rad])
                    min_val_2 = np.nanmin(this_row[rad+1:])
                    diff1 = self.tpw[jj,ii] - min_val_1
                    diff2 = self.tpw[jj,ii] - min_val_2
                    if (diff1 > thresh and diff2 > thresh):
                        is_local_max[jj,ii] = 1



            # print(f'nskip: {nskip}')
            dt22 = dt.datetime.now()
            print(('TPW Loop in AR Mask: ', (dt22 - dt11).total_seconds()))

            mask_tpw = is_local_max == 1
            """



            mask_tpw = ndimage.binary_fill_holes(mask_tpw)

            # Apply min. area criterion
            # label_im_tpw, nb_labels_tpw = ndimage_label_periodic_x(mask_tpw)
            # label_areas = ndimage.sum(area, label_im_tpw, range(nb_labels_tpw+1))

            # for ii in range(1, nb_labels_tpw+1):
            #     if label_areas[ii] < MINAREA:
            #         mask_tpw[label_im_tpw == ii] = 0

            self.mask_tpw = mask_tpw

            # Final AR mask is the combination of the IVT and TPW masks.
            # BUT I require that each feature have both a TPW and IVT mask feature.
            mask_combined = 1.0*np.logical_or(mask_ivt, mask_tpw)

            # TPW Mask MUST overlap with IVT mask to count.
            # mask_labeled, nb_labels = ndimage_label_periodic_x(mask_combined)
            # for n in range(nb_labels+1):
            #     if np.nansum(np.logical_and(mask_labeled==n, mask_ivt)) < 1:
            #         mask_combined[mask_labeled == n] = 0
            #     if np.nansum(np.logical_and(mask_labeled==n, mask_tpw)) < 1:
            #         mask_combined[mask_labeled == n] = 0

            # Remove blobs that are just too small.
            # Must have a size of at least the MINAREA
            mask_labeled, nb_labels = ndimage_label_periodic_x(mask_combined)
            for n in range(nb_labels+1):
                if np.nansum(area * (mask_labeled==n)) < MINAREA:
                    mask_combined[mask_labeled == n] = 0

            # Remove blobs that don't stick out far (MINAREA) from the deep_tropics_mask
            label_im_combined, nb_labels_combined = ndimage_label_periodic_x(mask_combined)
            for ii in range(1, nb_labels_combined+1):
                if np.nansum(area * np.logical_and(label_im_combined==ii, ~self.deep_tropics_mask)) < 0.5*MINAREA:
                    mask_combined[label_im_combined == ii] = 0


            self.mask = mask_combined
            self.mask = ndimage.binary_fill_holes(self.mask)

        else:
            raise Exception('Calculate TPW background and IVT background '
                            + 'before calling calc_ar_mask.')


    def __set_2d_field(self, field, data):
        """
        This low-level method is private
        because I really do not want the user to directly
        set things like grid_cell_area, ivt_eastward, and ivt_westward,
        which are derived from other variables.

        The method does apply a dimensions check to make sure:
        - Data coordinates (lat and lon) have been set.
        - The data dimensions are consistent with lat and lon dimensions.
        """
        if self.are_coordinates_set(): 
            if self.lon.shape == data.shape and self.lat.shape == data.shape:
                setattr(self, field, data)
            else:
                raise Exception(f"{field}: 2D data dimensions not consistent with lat/lon.")
        else:
            raise Exception("Set coordinates first!")


    def is_point_within_bounds(self, point_lon, point_lat):

        if self.are_coordinates_set(): 

            is_point_in_lon_bounds = (point_lon > np.min(self.lon) and point_lon < np.max(self.lon))
            is_point_in_lat_bounds = (point_lat > np.min(self.lat) and point_lat < np.max(self.lat))
            return (is_point_in_lon_bounds and is_point_in_lat_bounds) 

        else:
            raise Exception("Set coordinates first!")


    def is_point_in_feature(self, mask, point_lon, point_lat):
        rgi = RegularGridInterpolator(points=[self.lat[:,0], self.lon[0,:]], values=mask, method='nearest')
        return (rgi([point_lon, point_lat]) > 0.5)


    def get_initial_fg(self, this_label_mask):
        ivt_masked = 1.0 * self.ivt
        ivt_masked[this_label_mask < 0.5] = -999.0
        j0, i0 = np.unravel_index(np.argmax(ivt_masked), ivt_masked.shape)
        return (self.lon[j0,i0], self.lat[j0,i0])



    def get_weighted_ivt_angle(self, mask, lonc, latc, percentile=30.0):

        dist = np.sqrt(np.power(self.lon - lonc, 2) + np.power(self.lat - latc, 2))
        dist[mask < 0.5] = 9999.0

        dist_in_feature = dist[mask > 0.5]
        dist_cutoff = np.percentile(dist_in_feature, percentile)

        # weighted mean distance.
        dist_max = np.nanmax(dist[dist <= dist_cutoff])
        dist_min = np.nanmin(dist[dist <= dist_cutoff])
        w = (dist_max - dist) / (dist_max - dist_min)
        viwve_mean = np.nansum(w[dist <= dist_cutoff] * self.ivt_eastward[dist <= dist_cutoff]) / np.nansum(w[dist <= dist_cutoff])
        viwvn_mean = np.nansum(w[dist <= dist_cutoff] * self.ivt_northward[dist <= dist_cutoff]) / np.nansum(w[dist <= dist_cutoff])

        theta = np.arctan2(viwvn_mean, viwve_mean)

        # Get fraction ahead.
        lon1 = self.lon[dist <= dist_cutoff]
        lat1 = self.lat[dist <= dist_cutoff]
        dot_prod_keep = np.array([np.dot([lon1[jj] - lonc, lat1[jj] - latc], [np.cos(np.pi*theta/180.0), np.sin(np.pi*theta/180.0)]) for jj in range(len(lon1))])

        frac_ahead = np.sum(dot_prod_keep > 0.0) / len(dot_prod_keep)
        frac_ahead = min(frac_ahead, 1.0-frac_ahead)

        return theta, frac_ahead


    def get_weighted_ar_pathway_location(self, mask, lonc, latc, theta):

        # NOTE: Theta here is ALONG the axis of the feature.

        # Cross section.
        dxy = np.arange(-20.0, 20.1, 0.1)
        x_xc = lonc + np.cos(theta+np.pi/2) * dxy
        y_xc = latc + np.sin(theta+np.pi/2) * dxy

        # Make sure I don't cross 0 deg. meridian
        keep = [self.is_point_within_bounds(x_xc[ii], y_xc[ii]) for ii in range(len(x_xc))]
        x_xc = x_xc[keep]
        y_xc = y_xc[keep]

        rgi = RegularGridInterpolator(points=[self.lat[:,0], self.lon[0,:]], values=self.ivt)
        ivt_xc = rgi(np.array([y_xc, x_xc]).T)

        rgi = RegularGridInterpolator(points=[self.lat[:,0], self.lon[0,:]], values=mask, method='nearest')
        this_label_mask_xc = rgi(np.array([y_xc, x_xc]).T)

        # keep = (this_label_mask_xc > 0.5)  # This would take the entire feature

        label_im, nb_labels = ndimage.label(this_label_mask_xc)  # Take the largest contiguous feature.
        sizes = np.array(ndimage.sum(1, label_im, range(nb_labels+1)))
        sizes[0] = -1        
        label = np.argmax(sizes)
        keep = label_im == label

        if np.sum(keep) > 0:
            x_xc_keep = x_xc[keep]
            y_xc_keep = y_xc[keep]
            ivt_xc_keep = ivt_xc[keep]

            # IVT Weights
            weights_ivt = (ivt_xc_keep - np.min(ivt_xc_keep)) / (np.max(ivt_xc_keep) - np.min(ivt_xc_keep))

            # Distance Weights
            dist_xc = np.sqrt(np.power(x_xc_keep - lonc, 2) + np.power(y_xc_keep - latc, 2))
            weights_dist = (np.max(dist_xc) - dist_xc) / (np.max(dist_xc) - np.min(dist_xc))

            # Combine to get IVT + distance weights.
            weights = np.sqrt(np.power(weights_ivt,2) + np.power(weights_dist,2))

            # Calculate weighted center.
            lon_pathway = np.sum(weights * x_xc_keep) / np.sum(weights)
            lat_pathway = np.sum(weights * y_xc_keep) / np.sum(weights)

            # width = np.nanmax(111.0 * np.cos(np.pi * lat_pathway / 180.0) * dist_xc)
            width = np.sqrt(
                np.power(111.0 * np.cos(np.pi*np.mean(y_xc_keep)/180.0) * (np.max(x_xc_keep)-np.min(x_xc_keep)), 2)
                + np.power(111.0*(np.max(y_xc_keep)-np.min(y_xc_keep)), 2))

            
        else:
            lon_pathway = np.nan
            lat_pathway = np.nan
            width = np.nan

        return (lon_pathway, lat_pathway, width)


    def ar_axis_and_width(self, label_im, label, verbose=False):

        DXY_PATHWAY = 1.0
        TTMAX = 200 # 1000 # Max iterations forward or backwards
        FRAC_BEHIND = 0.1 # Min. frac. points that are "in front" of the system.

        if verbose:
            print(f'Doing central axis: {label} of {np.nanmax(label_im)}.')

        this_label_mask = 1.0*(label_im == label)
        npts = np.sum(label_im == label)


        # Initial first guess: XC through the max IVT within the feature.
        x_fg0, y_fg0 = self.get_initial_fg(this_label_mask)
        theta_next, frac_ahead = self.get_weighted_ivt_angle(this_label_mask, x_fg0, y_fg0)
        if verbose:
            print(f'Fraction ahead: {frac_ahead}')
        lon_pathway_next, lat_pathway_next, width_next = self.get_weighted_ar_pathway_location(this_label_mask, x_fg0, y_fg0, theta_next)

        lon_pathway_collect = [lon_pathway_next,]
        lat_pathway_collect = [lat_pathway_next,]
        theta_collect = [theta_next,]
        width_collect = [width_next,]

        for tt in range(TTMAX):
            x_fg = lon_pathway_collect[-1] - DXY_PATHWAY * np.cos(theta_collect[-1])
            y_fg = lat_pathway_collect[-1] - DXY_PATHWAY * np.sin(theta_collect[-1])
            if not self.is_point_within_bounds(x_fg, y_fg):
                if verbose:
                    print('Backwards: Reached the edge of the tracking domain.')
                break

            theta_next, frac_ahead = self.get_weighted_ivt_angle(this_label_mask, x_fg, y_fg)
            if verbose:
                print(f'Fraction ahead: {frac_ahead}')
            if frac_ahead < 0.1:
                if verbose:
                    print('Backwards: Reached frac. ahead < 0.1.')
                break
            lon_pathway_next, lat_pathway_next, width_next = self.get_weighted_ar_pathway_location(this_label_mask, x_fg, y_fg, theta_next)
            if np.isnan(lon_pathway_next):
                if verbose:
                    print('Backwards: Reached the end of the feature.')
                break
            if not self.is_point_within_bounds(lon_pathway_next, lat_pathway_next):
                if verbose:
                    print('Backwards: Reached the edge of the tracking domain.')
                break

            ## If I got too close to a previous point, stop.
            dist_from_other_points = np.sqrt(
                np.power(np.array(lon_pathway_collect) - lon_pathway_next, 2)
                + np.power(np.array(lat_pathway_collect) - lat_pathway_next, 2)
            )

            dist_from_last_point = np.sqrt(
                np.power(lon_pathway_collect[-1] - lon_pathway_next, 2)
                + np.power(lat_pathway_collect[-1] - lat_pathway_next, 2)
            )

            if np.nanmin(dist_from_other_points) > 0.5 and dist_from_last_point < 2.0:

                lon_pathway_collect += [lon_pathway_next,]
                lat_pathway_collect += [lat_pathway_next,]
                theta_collect += [theta_next,]
                width_collect += [width_next,]

            else:
                if verbose:
                    print('Backwards: looped back around.')
                break

        # Because I was going backwards, flip the order.
        # Then go forwards.
        lon_pathway_collect = lon_pathway_collect[::-1]
        lat_pathway_collect = lat_pathway_collect[::-1]
        theta_collect = theta_collect[::-1]
        width_collect = width_collect[::-1]

        # No need to re-inialize forward tracking. I will append to the backwards tracking result.
        for tt in range(TTMAX):
            
            x_fg = lon_pathway_collect[-1] + DXY_PATHWAY * np.cos(theta_collect[-1])
            y_fg = lat_pathway_collect[-1] + DXY_PATHWAY * np.sin(theta_collect[-1])
            if not self.is_point_within_bounds(x_fg, y_fg):
                if verbose:
                    print('Forward:   Reached the edge of the tracking domain.')
                break

            theta_next, frac_ahead = self.get_weighted_ivt_angle(this_label_mask, x_fg, y_fg)
            if verbose:
                print(f'Fraction ahead: {frac_ahead}')
            if frac_ahead < 0.1:
                if verbose:
                    print('Forward: Reached frac. ahead < 0.1.')
                break
            lon_pathway_next, lat_pathway_next, width_next = self.get_weighted_ar_pathway_location(this_label_mask, x_fg, y_fg, theta_next)
            if np.isnan(lon_pathway_next):
                if verbose:
                    print('Forward:   Reached the end of the feature.')
                break
            if not self.is_point_within_bounds(lon_pathway_next, lat_pathway_next):
                if verbose:
                    print('Forward:   Reached the edge of the tracking domain.')
                break

            ## If I got too close to a previous point, stop.
            dist_from_other_points = np.sqrt(
                np.power(np.array(lon_pathway_collect) - lon_pathway_next, 2)
                + np.power(np.array(lat_pathway_collect) - lat_pathway_next, 2)
            )

            dist_from_last_point = np.sqrt(
                np.power(lon_pathway_collect[-1] - lon_pathway_next, 2)
                + np.power(lat_pathway_collect[-1] - lat_pathway_next, 2)
            )

            if np.nanmin(dist_from_other_points) > 0.5 and dist_from_last_point < 2.0:

                lon_pathway_collect += [lon_pathway_next,]
                lat_pathway_collect += [lat_pathway_next,]
                theta_collect += [theta_next,]
                width_collect += [width_next,]

            else:
                if verbose:
                    print('Forward: looped back around.')
                break

        # list --> Numpy arrays
        lon_pathway_collect = np.array(lon_pathway_collect)
        lat_pathway_collect = np.array(lat_pathway_collect)
        theta_collect = np.array(theta_collect)
        width_collect = np.array(width_collect)

        # Make sure the path goes from west to east.
        if lon_pathway_collect[-1] < lon_pathway_collect[0]:
            lon_pathway_collect = lon_pathway_collect[::-1]
            lat_pathway_collect = lat_pathway_collect[::-1]
            theta_collect = theta_collect[::-1]
            width_collect = width_collect[::-1]

        # Derive some metrics.
        length = 0.0
        for ii in range(1, len(lon_pathway_collect)):
            length += np.sqrt(
                np.power(111.0 * np.cos(np.pi*lat_pathway_collect[ii]/180.0) * (lon_pathway_collect[ii]-lon_pathway_collect[ii-1]), 2)
                + np.power(111.0*(lat_pathway_collect[ii]-lat_pathway_collect[ii-1]), 2))

        # Take out any values that were set to NaN.
        keep = np.isfinite(lon_pathway_collect)
        lon_pathway_collect = lon_pathway_collect[keep]
        lat_pathway_collect = lat_pathway_collect[keep]
        width_collect = width_collect[keep]

        if verbose:
            print(f'length = {length} km.')
            print('mean width: {} km.'.format(np.nanmean(width_collect)))
            print('max width:  {} km.'.format(np.nanmax(width_collect)))
            print('length/width ratio:  {0:.2f}.'.format(length / np.nanmean(width_collect)))

        return (length, lon_pathway_collect, lat_pathway_collect, width_collect, theta_collect)
